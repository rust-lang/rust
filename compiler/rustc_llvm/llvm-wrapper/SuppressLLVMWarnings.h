#ifndef _rustc_llvm_SuppressLLVMWarnings_h
#define _rustc_llvm_SuppressLLVMWarnings_h

// LLVM currently generates many warnings when compiled using MSVC. These
// warnings make it difficult to diagnose real problems when working on C++
// code, so we suppress them.

#ifdef _MSC_VER
#pragma warning(disable : 4530) // C++ exception handler used, but unwind
                                // semantics are not enabled.
#pragma warning(                                                               \
    disable : 4624) // 'xxx': destructor was implicitly defined as deleted
#pragma warning(                                                               \
    disable : 4244) // conversion from 'xxx' to 'yyy', possible loss of data
#endif

#endif // _rustc_llvm_SuppressLLVMWarnings_h
