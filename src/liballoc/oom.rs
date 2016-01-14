// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::sync::atomic::{AtomicPtr, Ordering};
use core::mem;
use core::intrinsics;

static OOM_HANDLER: AtomicPtr<()> = AtomicPtr::new(default_oom_handler as *mut ());

fn default_oom_handler() -> ! {
    // The default handler can't do much more since we can't assume the presence
    // of libc or any way of printing an error message.
    unsafe { intrinsics::abort() }
}

/// Common out-of-memory routine
#[cold]
#[inline(never)]
#[unstable(feature = "oom", reason = "not a scrutinized interface",
           issue = "27700")]
pub fn oom() -> ! {
    let value = OOM_HANDLER.load(Ordering::SeqCst);
    let handler: fn() -> ! = unsafe { mem::transmute(value) };
    handler();
}

/// Set a custom handler for out-of-memory conditions
///
/// To avoid recursive OOM failures, it is critical that the OOM handler does
/// not allocate any memory itself.
#[unstable(feature = "oom", reason = "not a scrutinized interface",
           issue = "27700")]
pub fn set_oom_handler(handler: fn() -> !) {
    OOM_HANDLER.store(handler as *mut (), Ordering::SeqCst);
}
