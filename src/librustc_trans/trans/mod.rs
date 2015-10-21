// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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

pub use self::base::trans_crate;
pub use self::context::CrateContext;
pub use self::common::gensym_name;

#[macro_use]
mod macros;

mod adt;
mod asm;
mod attributes;
mod base;
mod basic_block;
mod build;
mod builder;
mod cabi;
mod cabi_aarch64;
mod cabi_arm;
mod cabi_mips;
mod cabi_powerpc;
mod cabi_x86;
mod cabi_x86_64;
mod cabi_x86_win64;
mod callee;
mod cleanup;
mod closure;
mod common;
mod consts;
mod context;
mod controlflow;
mod datum;
mod debuginfo;
mod declare;
mod expr;
mod foreign;
mod glue;
mod inline;
mod intrinsic;
mod llrepr;
mod machine;
mod _match;
mod meth;
mod mir;
mod monomorphize;
mod tvec;
mod type_;
mod type_of;
mod value;

#[derive(Copy, Clone)]
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
    pub no_builtins: bool,
}
