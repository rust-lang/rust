// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate provides static instrumentation macros.
//!
//! With the `probe!` macro, programmers can place static instrumentation
//! points in their code to mark events of interest.  These are compiled into
//! platform-specific implementations, e.g. SystemTap SDT on Linux.  Probes are
//! designed to have negligible overhead during normal operation, so they can
//! be present in all builds, and only activated using those external tools.
//!
//! # Example
//!
//! This simple example instruments the beginning and end of program, as well
//! as every iteration through the loop with arguments for the counter and
//! intermediate total.
//!
//! ```rust
//! #![feature(phase)]
//! #[phase(syntax)]
//! extern crate probe;
//! fn main() {
//!     probe!(foo, begin);
//!     let mut total = 0;
//!     for i in range(0, 100) {
//!         total += i;
//!         probe!(foo, loop, i, total);
//!     }
//!     assert_eq!(total, 4950);
//!     probe!(foo, end);
//! }
//! ```
//!
//! ## Using probes with SystemTap
//!
//! For the program above, a SystemTap script could double-check the totals:
//!
//! ```notrust
//! global check
//!
//! probe process.provider("foo").mark("loop") {
//!     check += $arg1;
//!     if (check != $arg2)
//!         printf("foo total is out of sync! (%d != %d)\n", check, $arg2);
//! }
//!
//! // .provider is optional
//! probe process.mark("begin"), process.mark("end") {
//!     printf("%s:%s\n", $$provider, $$name);
//! }
//! ```
//!
//! Since this program behaves as expected, this script will not have any complaint.
//!
//! ```notrust
//! $ stap --dyninst foo.stp -c ./foo
//! foo:begin
//! foo:end
//! ```
//!
//! ## Using probes with GDB
//!
//! Starting in version 7.5, GDB can set breakpoints on probes and read arguments.
//!
//! ```notrust
//! (gdb) info probes
//! Provider Name  Where              Semaphore Object
//! foo      begin 0x0000000000402e70           /tmp/foo
//! foo      end   0x000000000040315c           /tmp/foo
//! foo      loop  0x0000000000402f25           /tmp/foo
//! (gdb) break -probe foo:loop
//! Breakpoint 1 at 0x402f25
//! (gdb) condition 1 $_probe_arg1 > 1000
//! (gdb) run
//! Starting program: /tmp/foo
//! [Thread debugging using libthread_db enabled]
//! Using host libthread_db library "/lib64/libthread_db.so.1".
//!
//! Breakpoint 1, 0x0000000000402f25 in main::hd67360886023c1c6faa::v0.0 ()
//! (gdb) print $_probe_arg0
//! $1 = 45
//! (gdb) print $_probe_arg1
//! $2 = 1035
//! ```

#![crate_id = "probe#0.11-pre"]
#![crate_type = "dylib"]
#![experimental]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

#![feature(asm, macro_rules)]

mod platform;

/// Define a static probe point.
///
/// This annotates a code location with a name and arguments, and compiles
/// in metadata to let debugging tools locate it.
///
/// # Arguments
///
/// * `provider` - An identifier for naming probe groups.
///
/// * `name`     - An identifier for this specific probe.
///
/// * `arg`...  - Optional data to provide with the probe.  Any expression which
///   can be cast `as i64` is allowed as an argument.  The arguments might not
///   be evaluated at all when a debugger is not attached to the probe,
///   depending on the platform implementation, so don't rely on side effects.
///
/// # Example
///
/// ```
/// #![feature(phase)]
/// #[phase(syntax)]
/// extern crate probe;
/// fn main() {
///     probe!(foo, main);
///
///     let x = 42;
///     probe!(foo, show_x, x);
///
///     let y = Some(x);
///     probe!(foo, show_y, match y {
///         Some(n) => n,
///         None    => -1
///     });
/// }
/// ```
#[macro_export]
macro_rules! probe(
    ($provider:ident, $name:ident $(, $arg:expr)*)
    => (platform_probe!($provider, $name $(, $arg)*));
)
