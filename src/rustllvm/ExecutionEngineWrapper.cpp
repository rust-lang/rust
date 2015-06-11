// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "rustllvm.h"

#include "llvm/ExecutionEngine/SectionMemoryManager.h"

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::object;

// libmorestack is not used on other platforms
#if defined(__linux__) || defined(__APPLE__)
extern "C" void __morestack(void);

static void* morestack_addr() {
    return reinterpret_cast<void*>(__morestack);
}
#endif

class RustJITMemoryManager : public SectionMemoryManager
{
    typedef SectionMemoryManager Base;

    public:

    RustJITMemoryManager() {}

    uint64_t getSymbolAddress(const std::string &Name) override
    {
#if defined(__linux__) || defined(__APPLE__)
        if (Name == "__morestack" || Name == "___morestack")
            return reinterpret_cast<uint64_t>(__morestack);
        if (Name == "__morestack_addr" || Name == "___morestack_addr")
            return reinterpret_cast<uint64_t>(morestack_addr);
#endif

        return Base::getSymbolAddress(Name);
    }
};

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(RustJITMemoryManager, LLVMRustJITMemoryManagerRef)

extern "C" LLVMBool LLVMRustLoadDynamicLibrary(const char *path)
{
    std::string err;
    DynamicLibrary lib = DynamicLibrary::getPermanentLibrary(path, &err);

    if (!lib.isValid())
        LLVMRustSetLastError(err.c_str());

    return lib.isValid();
}

// Calls LLVMAddModule;
// exists for consistency with LLVMExecutionEngineRemoveModule
extern "C" void LLVMExecutionEngineAddModule(
    LLVMExecutionEngineRef eeref, LLVMModuleRef mref)
{
#ifdef _WIN32
    // On Windows, MCJIT must generate ELF objects
    std::string target = getProcessTriple();
    target += "-elf";
    target = Triple::normalize(target);
    unwrap(mref)->setTargetTriple(target);
#endif
    LLVMAddModule(eeref, mref);
}

// LLVMRemoveModule exists in LLVM's C bindings,
// but it requires pointless parameters
extern "C" LLVMBool LLVMExecutionEngineRemoveModule(
    LLVMExecutionEngineRef eeref, LLVMModuleRef mref)
{
    ExecutionEngine *ee = unwrap(eeref);
    Module *m = unwrap(mref);

    return ee->removeModule(m);
}

extern "C" LLVMExecutionEngineRef LLVMBuildExecutionEngine(LLVMModuleRef mod)
{
    // These are necessary for code generation to work properly.
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

#ifdef _WIN32
    // On Windows, MCJIT must generate ELF objects
    std::string target = getProcessTriple();
    target += "-elf";
    target = Triple::normalize(target);
    unwrap(mod)->setTargetTriple(target);
#endif

    std::string error_str;
    TargetOptions options;

    options.JITEmitDebugInfo = true;
    options.NoFramePointerElim = true;

    RustJITMemoryManager *mm = new RustJITMemoryManager;

    ExecutionEngine *ee =
    #if LLVM_VERSION_MINOR >= 6
        EngineBuilder(std::unique_ptr<Module>(unwrap(mod)))
            .setMCJITMemoryManager(std::unique_ptr<RustJITMemoryManager>(mm))
    #else
        EngineBuilder(unwrap(mod))
            .setMCJITMemoryManager(mm)
    #endif
            .setEngineKind(EngineKind::JIT)
            .setErrorStr(&error_str)
            .setTargetOptions(options)
            .create();

    if (!ee)
        LLVMRustSetLastError(error_str.c_str());

    return wrap(ee);
}

extern "C" void LLVMExecutionEngineFinalizeObject(LLVMExecutionEngineRef eeref)
{
    ExecutionEngine *ee = unwrap(eeref);

    ee->finalizeObject();
}
