// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that item-less traits do not cause dropck to inject extra
// region constraints.

#![allow(non_camel_case_types)]

#![feature(dropck_parametricity)]

trait UserDefined { }

impl UserDefined for i32 { }
impl<'a, T> UserDefined for &'a T { }

// e.g. `impl_drop!(Send, D_Send)` expands to:
//   ```rust
//   struct D_Send<T:Send>(T);
//   impl<T:Send> Drop for D_Send<T> { fn drop(&mut self) { } }
//   ```
macro_rules! impl_drop {
    ($Bound:ident, $Id:ident) => {
        struct $Id<T:$Bound>(T);
        impl <T:$Bound> Drop for $Id<T> {
            #[unsafe_destructor_blind_to_params]
            fn drop(&mut self) { }
        }
    }
}

impl_drop!{Send,         D_Send}
impl_drop!{Sized,        D_Sized}

// See note below regarding Issue 24895
// impl_drop!{Copy,         D_Copy}

impl_drop!{Sync,         D_Sync}
impl_drop!{UserDefined,  D_UserDefined}

macro_rules! body {
    ($id:ident) => { {
        // `_d` and `d1` are assigned the *same* lifetime by region inference ...
        let (_d, d1);

        d1 = $id(1);
        // ... we store a reference to `d1` within `_d` ...
        _d = $id(&d1);

        // ... a *conservative* dropck will thus complain, because it
        // thinks Drop of _d could access the already dropped `d1`.
    } }
}

fn f_send() { body!(D_Send) }
fn f_sized() { body!(D_Sized) }
fn f_sync() { body!(D_Sync) }

// Issue 24895: Copy: Clone implies `impl<T:Copy> Drop for ...` can
// access a user-defined clone() method, which causes this test case
// to fail.
//
// If 24895 is resolved by removing the `Copy: Clone` relationship,
// then this definition and the call below should be uncommented. If
// it is resolved by deciding to keep the `Copy: Clone` relationship,
// then this comment and the associated bits of code can all be
// removed.

// fn f_copy() { body!(D_Copy) }

fn f_userdefined() { body!(D_UserDefined) }

fn main() {
    f_send();
    f_sized();
    // See note above regarding Issue 24895.
    // f_copy();
    f_sync();
    f_userdefined();
}
