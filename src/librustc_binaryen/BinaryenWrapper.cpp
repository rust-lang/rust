// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a small C API inserted on top of the Binaryen C++ API which we use
// from Rust. Once we have a real linker for we'll be able to remove all this,
// and otherwise this is just all on a "as we need it" basis for now.

#include <stdint.h>
#include <string>
#include <sstream>
#include <stdlib.h>

#include "s2wasm.h"
#include "wasm-binary.h"
#include "wasm-linker.h"

using namespace wasm;

struct BinaryenRustModule {
  BufferWithRandomAccess buffer;
  std::string sourceMapJSON;
};

struct BinaryenRustModuleOptions {
  uint64_t globalBase;
  bool debug;
  uint64_t stackAllocation;
  uint64_t initialMem;
  uint64_t maxMem;
  bool importMemory;
  bool ignoreUnknownSymbols;
  bool debugInfo;
  std::string startFunction;
  std::string sourceMapUrl;

  BinaryenRustModuleOptions() :
    globalBase(0),
    debug(false),
    stackAllocation(0),
    initialMem(0),
    maxMem(0),
    importMemory(false),
    ignoreUnknownSymbols(false),
    debugInfo(false),
    startFunction(""),
    sourceMapUrl("")
  {}

};

extern "C" BinaryenRustModuleOptions*
BinaryenRustModuleOptionsCreate() {
  return new BinaryenRustModuleOptions;
}

extern "C" void
BinaryenRustModuleOptionsFree(BinaryenRustModuleOptions *options) {
  delete options;
}

extern "C" void
BinaryenRustModuleOptionsSetDebugInfo(BinaryenRustModuleOptions *options,
                                      bool debugInfo) {
  options->debugInfo = debugInfo;
}

extern "C" void
BinaryenRustModuleOptionsSetStart(BinaryenRustModuleOptions *options,
                                  char *start) {
  options->startFunction = start;
}

extern "C" void
BinaryenRustModuleOptionsSetSourceMapUrl(BinaryenRustModuleOptions *options,
                                         char *sourceMapUrl) {
  options->sourceMapUrl = sourceMapUrl;
}

extern "C" void
BinaryenRustModuleOptionsSetStackAllocation(BinaryenRustModuleOptions *options,
                                            uint64_t stack) {
  options->stackAllocation = stack;
}

extern "C" void
BinaryenRustModuleOptionsSetImportMemory(BinaryenRustModuleOptions *options,
                                         bool import) {
  options->importMemory = import;
}

extern "C" BinaryenRustModule*
BinaryenRustModuleCreate(const BinaryenRustModuleOptions *options,
                         const char *assembly) {
  Linker linker(
      options->globalBase,
      options->stackAllocation,
      options->initialMem,
      options->maxMem,
      options->importMemory,
      options->ignoreUnknownSymbols,
      options->startFunction,
      options->debug);

  S2WasmBuilder mainbuilder(assembly, options->debug);
  linker.linkObject(mainbuilder);
  linker.layout();

  auto ret = make_unique<BinaryenRustModule>();
  {
    WasmBinaryWriter writer(&linker.getOutput().wasm, ret->buffer, options->debug);
    writer.setNamesSection(options->debugInfo);

    std::unique_ptr<std::ostringstream> sourceMapStream = nullptr;
    {
      sourceMapStream = make_unique<std::ostringstream>();
      writer.setSourceMap(sourceMapStream.get(), options->sourceMapUrl);
    }

    // FIXME: support symbol maps?
    // writer.setSymbolMap(symbolMap);
    writer.write();

    if (sourceMapStream) {
      ret->sourceMapJSON = sourceMapStream->str();
    }
  }
  return ret.release();
}

extern "C" const uint8_t*
BinaryenRustModulePtr(const BinaryenRustModule *M) {
  return M->buffer.data();
}

extern "C" size_t
BinaryenRustModuleLen(const BinaryenRustModule *M) {
  return M->buffer.size();
}

extern "C" const char*
BinaryenRustModuleSourceMapPtr(const BinaryenRustModule *M) {
  return M->sourceMapJSON.data();
}

extern "C" size_t
BinaryenRustModuleSourceMapLen(const BinaryenRustModule *M) {
  return M->sourceMapJSON.length();
}

extern "C" void
BinaryenRustModuleFree(BinaryenRustModule *M) {
  delete M;
}
