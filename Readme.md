# The Enzyme Automatic differentiator

To build Enzyme run the following steps:
```bash
cd enzyme && mkdir build && cd build
cmake ..
make
```

This will pick up your default installation of LLVM, to specify a specific LLVM installation add the LLVM_DIR flag to cmake as follows:

```bash
cmake .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

To run Enzyme tests use the following:
```bash
make check-enzyme
```

If you run the above and get an error like `/bin/sh: 1: ../../: Permission denied` or ` ../../ not found`, it's likely that cmake wasn't able to find your version of llvm-lit, LLVM's unit tester. This often happens if you use the default Ubuntu install of LLVM as they stopped including it in their packaging. To remedy, find lit.py or lit or llvm-lit on your system and add the following flag to cmake:
```bash
cmake .. -DLLVM_EXTERNAL_LIT=/path/to/lit/lit.py
```

To run C++ integration tests (which require boost/eigen and take quite a while) use the following:
```bash
make check-enzyme-integration
```
