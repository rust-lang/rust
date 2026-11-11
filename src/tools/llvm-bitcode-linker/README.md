# LLVM Bitcode Linker
The LLVM bitcode linker can be used to link targets without any dependency on system libraries.
The code will be linked in llvm-bc before compiling to native code. For some of these targets
(e.g. ptx) there does not exist a sensible way to link the native format at all. A bitcode linker
is required to link code compiled for such targets.
