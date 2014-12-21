// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{ContextRef, ModuleRef};
use metadata::common::LinkMeta;
use middle::dependency_format;

pub use self::base::trans_crate;
pub use self::context::CrateContext;
pub use self::common::gensym_name;

mod doc;
mod macros;
mod inline;
mod monomorphize;
mod controlflow;
mod glue;
mod datum;
mod callee;
mod expr;
mod common;
mod context;
mod consts;
mod type_of;
mod build;
mod builder;
mod base;
mod _match;
mod closure;
mod tvec;
mod meth;
mod cabi;
mod cabi_x86;
mod cabi_x86_64;
mod cabi_x86_win64;
mod cabi_arm;
mod cabi_mips;
mod foreign;
mod intrinsic;
mod debuginfo;
mod machine;
mod adt;
mod asm;
mod type_;
mod value;
mod basic_block;
mod llrepr;
mod cleanup;

#[deriving(Copy)]
pub struct ModuleTranslation {
    pub llcx: ContextRef,
    pub llmod: ModuleRef,
}

unsafe impl Send for ModuleTranslation { }
unsafe impl Sync for ModuleTranslation { }

pub struct CrateTranslation {
    pub modules: Vec<ModuleTranslation>,
    pub metadata_module: ModuleTranslation,
    pub link: LinkMeta,
    pub metadata: Vec<u8>,
    pub reachable: Vec<String>,
    pub crate_formats: dependency_format::Dependencies,
    pub no_builtins: bool,
}

