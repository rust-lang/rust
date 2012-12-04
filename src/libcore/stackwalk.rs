// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)]; // FIXME #3538

#[legacy_modes]; // tjc: remove after snapshot

// NB: transitionary, de-mode-ing.
// XXX: Can't forbid this because frame_address needs a deprecated mode.
#[allow(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast::reinterpret_cast;
use ptr::offset;
use sys::size_of;

pub type Word = uint;

pub struct Frame {
    fp: *Word
}

pub fn Frame(fp: *Word) -> Frame {
    Frame {
        fp: fp
    }
}

pub fn walk_stack(visit: fn(Frame) -> bool) {

    debug!("beginning stack walk");

    do frame_address |frame_pointer| {
        let mut frame_address: *Word = unsafe {
            reinterpret_cast(&frame_pointer)
        };
        loop {
            let fr = Frame(frame_address);

            debug!("frame: %x", unsafe { reinterpret_cast(&fr.fp) });
            visit(fr);

            unsafe {
                let next_fp: **Word = reinterpret_cast(&frame_address);
                frame_address = *next_fp;
                if *frame_address == 0u {
                    debug!("encountered task_start_wrapper. ending walk");
                    // This is the task_start_wrapper_frame. There is
                    // no stack beneath it and it is a foreign frame.
                    break;
                }
            }
        }
    }
}

#[test]
fn test_simple() {
    for walk_stack |_frame| {
    }
}

#[test]
fn test_simple_deep() {
    fn run(i: int) {
        if i == 0 { return }

        for walk_stack |_frame| {
            unsafe {
                breakpoint();
            }
        }
        run(i - 1);
    }

    run(10);
}

fn breakpoint() {
    rustrt::rust_dbg_breakpoint()
}

fn frame_address(f: fn(++x: *u8)) {
    rusti::frame_address(f)
}

extern mod rustrt {
    #[legacy_exports];
    fn rust_dbg_breakpoint();
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn frame_address(f: fn(++x: *u8));
}
