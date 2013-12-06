// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session;
use middle::trans::base;
use middle::trans::type_::Type;
use lib::llvm::{ModuleRef, ValueRef};

pub struct Upcalls {
    rust_personality: ValueRef,
}

macro_rules! upcall (
    (nothrow fn $name:ident -> $ret:expr) => ({
        let fn_ty = Type::func([], &$ret);
        let decl = base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty);
        base::set_no_unwind(decl);
        decl
    })
)

pub fn declare_upcalls(_targ_cfg: @session::config,
                       llmod: ModuleRef) -> @Upcalls {
    @Upcalls {
        rust_personality: upcall!(nothrow fn rust_personality -> Type::i32()),
    }
}
