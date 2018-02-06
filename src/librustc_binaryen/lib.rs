// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rustc bindings to the binaryen project.
//!
//! This crate is a small shim around the binaryen project which provides us the
//! ability to take LLVM's output and generate a wasm module. Specifically this
//! only supports one operation, creating a module from LLVM's assembly format
//! and then serializing that module to a wasm module.

extern crate libc;

use std::slice;
use std::ffi::{CString, CStr};

/// In-memory representation of a serialized wasm module.
pub struct Module {
    ptr: *mut BinaryenRustModule,
}

impl Module {
    /// Creates a new wasm module from the LLVM-assembly provided (in a C string
    /// format).
    ///
    /// The actual module creation can be tweaked through the various options in
    /// `ModuleOptions` as well. Any errors are just returned as a bland string.
    pub fn new(assembly: &CStr, opts: &ModuleOptions) -> Result<Module, String> {
        unsafe {
            let ptr = BinaryenRustModuleCreate(opts.ptr, assembly.as_ptr());
            if ptr.is_null() {
                Err(format!("failed to create binaryen module"))
            } else {
                Ok(Module { ptr })
            }
        }
    }

    /// Returns the data of the serialized wasm module. This is a `foo.wasm`
    /// file contents.
    pub fn data(&self) -> &[u8] {
        unsafe {
            let ptr = BinaryenRustModulePtr(self.ptr);
            let len = BinaryenRustModuleLen(self.ptr);
            slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            BinaryenRustModuleFree(self.ptr);
        }
    }
}

pub struct ModuleOptions {
    ptr: *mut BinaryenRustModuleOptions,
}

impl ModuleOptions {
    pub fn new() -> ModuleOptions {
        unsafe {
            let ptr = BinaryenRustModuleOptionsCreate();
            ModuleOptions { ptr }
        }
    }

    /// Turns on or off debug info.
    ///
    /// From what I can tell this just creates a "names" section of the wasm
    /// module which contains a table of the original function names.
    pub fn debuginfo(&mut self, debug: bool) -> &mut Self {
        unsafe {
            BinaryenRustModuleOptionsSetDebugInfo(self.ptr, debug);
        }
        self
    }

    /// Configures a `start` function for the module, to be executed when it's
    /// loaded.
    pub fn start(&mut self, func: &str) -> &mut Self {
        let func = CString::new(func).unwrap();
        unsafe {
            BinaryenRustModuleOptionsSetStart(self.ptr, func.as_ptr());
        }
        self
    }

    /// Configures how much stack is initially allocated for the module. 1MB is
    /// probably good enough for now.
    pub fn stack(&mut self, amt: u64) -> &mut Self {
        unsafe {
            BinaryenRustModuleOptionsSetStackAllocation(self.ptr, amt);
        }
        self
    }

    /// Flags whether the initial memory should be imported or exported. So far
    /// we export it by default.
    pub fn import_memory(&mut self, import: bool) -> &mut Self {
        unsafe {
            BinaryenRustModuleOptionsSetImportMemory(self.ptr, import);
        }
        self
    }
}

impl Drop for ModuleOptions {
    fn drop(&mut self) {
        unsafe {
            BinaryenRustModuleOptionsFree(self.ptr);
        }
    }
}

enum BinaryenRustModule {}
enum BinaryenRustModuleOptions {}

extern {
    fn BinaryenRustModuleCreate(opts: *const BinaryenRustModuleOptions,
                                assembly: *const libc::c_char)
        -> *mut BinaryenRustModule;
    fn BinaryenRustModulePtr(module: *const BinaryenRustModule) -> *const u8;
    fn BinaryenRustModuleLen(module: *const BinaryenRustModule) -> usize;
    fn BinaryenRustModuleFree(module: *mut BinaryenRustModule);

    fn BinaryenRustModuleOptionsCreate()
        -> *mut BinaryenRustModuleOptions;
    fn BinaryenRustModuleOptionsSetDebugInfo(module: *mut BinaryenRustModuleOptions,
                                             debuginfo: bool);
    fn BinaryenRustModuleOptionsSetStart(module: *mut BinaryenRustModuleOptions,
                                         start: *const libc::c_char);
    fn BinaryenRustModuleOptionsSetStackAllocation(
        module: *mut BinaryenRustModuleOptions,
        stack: u64,
    );
    fn BinaryenRustModuleOptionsSetImportMemory(
        module: *mut BinaryenRustModuleOptions,
        import: bool,
    );
    fn BinaryenRustModuleOptionsFree(module: *mut BinaryenRustModuleOptions);
}
