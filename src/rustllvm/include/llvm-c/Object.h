/*===-- llvm-c/Object.h - Object Lib C Iface --------------------*- C++ -*-===*/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/*===----------------------------------------------------------------------===*/
/*                                                                            */
/* This header declares the C interface to libLLVMObject.a, which             */
/* implements object file reading and writing.                                */
/*                                                                            */
/* Many exotic languages can interoperate with C code but have a harder time  */
/* with C++ due to name mangling. So in addition to C, this interface enables */
/* tools written in such languages.                                           */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_OBJECT_H
#define LLVM_C_OBJECT_H

#include "llvm-c/Core.h"
#include "llvm/Config/llvm-config.h"

#ifdef __cplusplus
#include "llvm/Object/ObjectFile.h"

extern "C" {
#endif


typedef struct LLVMOpaqueObjectFile *LLVMObjectFileRef;

typedef struct LLVMOpaqueSectionIterator *LLVMSectionIteratorRef;

LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf);
void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile);

LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile);
void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI);
LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSectionIteratorRef SI);
void LLVMMoveToNextSection(LLVMSectionIteratorRef SI);
const char *LLVMGetSectionName(LLVMSectionIteratorRef SI);
uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI);
const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI);


#ifdef __cplusplus
}

namespace llvm {
  namespace object {
    inline ObjectFile *unwrap(LLVMObjectFileRef OF) {
      return reinterpret_cast<ObjectFile*>(OF);
    }

    inline LLVMObjectFileRef wrap(const ObjectFile *OF) {
      return reinterpret_cast<LLVMObjectFileRef>(const_cast<ObjectFile*>(OF));
    }

    inline ObjectFile::section_iterator *unwrap(LLVMSectionIteratorRef SI) {
      return reinterpret_cast<ObjectFile::section_iterator*>(SI);
    }

    inline LLVMSectionIteratorRef
    wrap(const ObjectFile::section_iterator *SI) {
      return reinterpret_cast<LLVMSectionIteratorRef>
        (const_cast<ObjectFile::section_iterator*>(SI));
    }
  }
}

#endif /* defined(__cplusplus) */

#endif

