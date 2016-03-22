// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use rustc::ty::Ty;
use adt;
use base;
use common::{self, BlockAndBuilder};
use machine;
use type_of;
use type_::Type;

pub fn drop_fill<'bcx, 'tcx>(bcx: &BlockAndBuilder<'bcx, 'tcx>, value: ValueRef, ty: Ty<'tcx>) {
    let llty = type_of::type_of(bcx.ccx(), ty);
    let llptr = bcx.pointercast(value, Type::i8(bcx.ccx()).ptr_to());
    let filling = common::C_u8(bcx.ccx(), adt::DTOR_DONE);
    let size = machine::llsize_of(bcx.ccx(), llty);
    let align = common::C_u32(bcx.ccx(), machine::llalign_of_min(bcx.ccx(), llty));
    base::call_memset(&bcx, llptr, filling, size, align, false);
}
