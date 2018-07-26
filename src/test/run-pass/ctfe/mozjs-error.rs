// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct CustomAutoRooterVFTable {
    trace: unsafe extern "C" fn(this: *mut i32, trc: *mut u32),
}

unsafe trait CustomAutoTraceable: Sized {
    const vftable: CustomAutoRooterVFTable = CustomAutoRooterVFTable {
        trace: Self::trace,
    };

    unsafe extern "C" fn trace(this: *mut i32, trc: *mut u32) {
        let this = this as *const Self;
        let this = this.as_ref().unwrap();
        Self::do_trace(this, trc);
    }

    fn do_trace(&self, trc: *mut u32);
}

unsafe impl CustomAutoTraceable for () {
    fn do_trace(&self, _: *mut u32) {
        // nop
    }
}

fn main() {
    let _ = <()>::vftable;
}
