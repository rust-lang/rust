//===- Object.cpp - C bindings to the object file library--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the C bindings to the file-format-independent object
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm-c/Object.h"

using namespace llvm;
using namespace object;

LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf) {
  return wrap(ObjectFile::createObjectFile(unwrap(MemBuf)));
}

void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile) {
  delete unwrap(ObjectFile);
}

LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile) {
  ObjectFile::section_iterator SI = unwrap(ObjectFile)->begin_sections();
  return wrap(new ObjectFile::section_iterator(SI));
}

void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSectionIteratorRef SI) {
  return (*unwrap(SI) == unwrap(ObjectFile)->end_sections()) ? 1 : 0;
}

void LLVMMoveToNextSection(LLVMSectionIteratorRef SI) {
  // We can't use unwrap() here because the argument to ++ must be an lvalue.
  ++*reinterpret_cast<ObjectFile::section_iterator*>(SI);
}

const char *LLVMGetSectionName(LLVMSectionIteratorRef SI) {
  return (*unwrap(SI))->getName().data();
}

uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI) {
  return (*unwrap(SI))->getSize();
}

const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI) {
  return (*unwrap(SI))->getContents().data();
}

