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
use middle::trans::type_::{Type, CrateTypes};
use lib::llvm::{ModuleRef, ValueRef};

pub struct Upcalls {
    trace: ValueRef,
    rust_personality: ValueRef,
    reset_stack_limit: ValueRef
}

macro_rules! upcall (
    (fn $name:ident($($arg:expr),+) -> $ret:expr) => ({
        let fn_ty = CrateTypes::func_([ $($arg),* ], &$ret);
        base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty)
    });
    (nothrow fn $name:ident($($arg:expr),+) -> $ret:expr) => ({
        let fn_ty = CrateTypes::func_([ $($arg),* ], &$ret);
        let decl = base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty);
        base::set_no_unwind(decl);
        decl
    });
    (nothrow fn $name:ident -> $ret:expr) => ({
        let fn_ty = CrateTypes::func_([], &$ret);
        let decl = base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty);
        base::set_no_unwind(decl);
        decl
    })
)

pub fn declare_upcalls(types: &CrateTypes, llmod: ModuleRef) -> @Upcalls {
    let opaque_ptr = types.i8p();
    let int_ty = types.i();

    @Upcalls {
        trace: upcall!(fn trace(opaque_ptr, opaque_ptr, int_ty) -> types.void()),
        rust_personality: upcall!(nothrow fn rust_personality -> types.i32()),
        reset_stack_limit: upcall!(nothrow fn reset_stack_limit -> types.void())
    }
}
