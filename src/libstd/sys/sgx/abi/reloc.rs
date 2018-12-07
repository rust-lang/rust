// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use slice::from_raw_parts;
use super::mem;

const R_X86_64_RELATIVE: u32 = 8;

#[repr(packed)]
struct Rela<T> {
    offset: T,
    info: T,
    addend: T,
}

pub fn relocate_elf_rela() {
    extern {
        static RELA: u64;
        static RELACOUNT: usize;
    }

    if unsafe { RELACOUNT } == 0 { return }  // unsafe ok: link-time constant

    let relas = unsafe {
        from_raw_parts::<Rela<u64>>(mem::rel_ptr(RELA), RELACOUNT)  // unsafe ok: link-time constant
    };
    for rela in relas {
        if rela.info != (/*0 << 32 |*/ R_X86_64_RELATIVE as u64) {
            panic!("Invalid relocation");
        }
        unsafe { *mem::rel_ptr_mut::<*const ()>(rela.offset) = mem::rel_ptr(rela.addend) };
    }
}
