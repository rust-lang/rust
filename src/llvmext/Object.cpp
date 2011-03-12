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

LLVMObjectFileRef LLVMCreateObjectFile(const char *ObjectPath) {
  StringRef SR(ObjectPath);
  return wrap(ObjectFile::createObjectFile(SR));
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

void LLVMMoveToNextSection(LLVMSectionIteratorRef SI) {
  ObjectFile::section_iterator UnwrappedSI = *unwrap(SI);
  ++UnwrappedSI;
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

