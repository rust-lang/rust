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
    trace: ValueRef,
    call_shim_on_c_stack: ValueRef,
    call_shim_on_rust_stack: ValueRef,
    rust_personality: ValueRef,
    reset_stack_limit: ValueRef
}

macro_rules! upcall (
    (fn $name:ident($($arg:expr),+) -> $ret:expr) => ({
        let fn_ty = Type::func([ $($arg),* ], &$ret);
        base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty)
    });
    (nothrow fn $name:ident($($arg:expr),+) -> $ret:expr) => ({
        let fn_ty = Type::func([ $($arg),* ], &$ret);
        let decl = base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty);
        base::set_no_unwind(decl);
        decl
    });
    (nothrow fn $name:ident -> $ret:expr) => ({
        let fn_ty = Type::func([], &$ret);
        let decl = base::decl_cdecl_fn(llmod, ~"upcall_" + stringify!($name), fn_ty);
        base::set_no_unwind(decl);
        decl
    })
)

pub fn declare_upcalls(targ_cfg: @session::config, llmod: ModuleRef) -> @Upcalls {
    let opaque_ptr = Type::i8().ptr_to();
    let int_ty = Type::int(targ_cfg.arch);

    @Upcalls {
        trace: upcall!(fn trace(opaque_ptr, opaque_ptr, int_ty) -> Type::void()),
        call_shim_on_c_stack: upcall!(fn call_shim_on_c_stack(opaque_ptr, opaque_ptr) -> int_ty),
        call_shim_on_rust_stack:
            upcall!(fn call_shim_on_rust_stack(opaque_ptr, opaque_ptr) -> int_ty),
        rust_personality: upcall!(nothrow fn rust_personality -> Type::i32()),
        reset_stack_limit: upcall!(nothrow fn reset_stack_limit -> Type::void())
    }
}
