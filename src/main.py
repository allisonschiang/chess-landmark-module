import asyncio
from viam.module.module import Module
from models.detector import Detector as DetectorModel


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
