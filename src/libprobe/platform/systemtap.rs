// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! SystemTap static probes
//!
//! This is a mechanism for developers to provide static annotations for
//! meaningful points in code, with arguments that indicate some relevant
//! state.  Such locations may be probed by SystemTap `process.mark("name")`,
//! and GDB can also locate them with `info probes` and `break -probe name`.
//!
//! The impact on code generation is designed to be minimal: just a single
//! `NOP` placeholder is added inline for instrumentation, and ELF notes
//! contain metadata to name the probe and describe the location of its
//! arguments.
//!
//! # Links:
//!
//! * https://sourceware.org/systemtap/man/stapprobes.3stap.html#lbAO (see `process.mark`)
//! * https://sourceware.org/systemtap/wiki/AddingUserSpaceProbingToApps
//! * https://sourceware.org/systemtap/wiki/UserSpaceProbeImplementation
//! * https://sourceware.org/gdb/onlinedocs/gdb/Static-Probe-Points.html

//
// DEVELOPER NOTES
//
// Arguments are currently type-casted as i64, because that directly maps to
// SystemTap's long, no matter the architecture.  However, if we could figure
// out types here, they could be annotated more specifically, for example an
// argstr of "4@$0 -2@$1" indicates u32 and i16 respectively.  Any pointer
// would be fine too, like *c_char, simply 4@ or 8@ for target_word_size.
//
// The macros in sdt.h don't know types either, so they split each argument
// into two asm inputs, roughly:
//   asm("[...]"
//       ".asciz \"%n[_SDT_S0]@%[_SDT_A0]\""
//       "[...]"
//     : :
//     [_SDT_S0] "n" ((_SDT_ARGSIGNED (x) ? 1 : -1) * (int) sizeof (x)),
//     [_SDT_A0] "nor" (x)
//     );
// where _SDT_ARGSIGNED is a macro using gcc builtins, so it's still resolved a
// compile time, and %n makes it a raw literal rather than an asm number.
//
// This might be a possible direction for Rust SDT to follow.  For LLVM
// InlineAsm, the string would look like "${0:n}@$1", but we need the size/sign
// for that first input, and that must be a numeric constant no matter what
// optimization level we're at.
//
// NB: If there were also a way to generate the positional "$0 $1 ..." indexes,
// then we could lose the manually-unrolled duplication below.  For now, expand
// up to 12 args, the same limit as sys/sdt.h.
//
// FIXME semaphores - SDT can define a short* that debuggers will increment when
// they attach, and decrement on detach.  Thus a probe_enabled!(provider,name)
// could return if that value != 0, to be used similarly to log_enabled!().  We
// could even be clever and skip argument evaluation altogether, the same way
// that log!() checks log_enabled!() first.
//

#[macro_export]
macro_rules! platform_probe(
    ($provider:ident, $name:ident)
    => (sdt_asm!($provider, $name, ""));

    ($provider:ident, $name:ident, $arg1:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0",
                 $arg1));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1",
                 $arg1, $arg2));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2",
                 $arg1, $arg2, $arg3));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3",
                 $arg1, $arg2, $arg3, $arg4));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4",
                 $arg1, $arg2, $arg3, $arg4, $arg5));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr, $arg8:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6 -8@$7",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr, $arg8:expr, $arg9:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6 -8@$7 -8@$8",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8, $arg9));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr, $arg8:expr, $arg9:expr, $arg10:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6 -8@$7 -8@$8 -8@$9",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8, $arg9, $arg10));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr, $arg8:expr, $arg9:expr, $arg10:expr, $arg11:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6 -8@$7 -8@$8 -8@$9 -8@$10",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8, $arg9, $arg10, $arg11));

    ($provider:ident, $name:ident, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr,
     $arg6:expr, $arg7:expr, $arg8:expr, $arg9:expr, $arg10:expr, $arg11:expr, $arg12:expr)
    => (sdt_asm!($provider, $name,
                 "-8@$0 -8@$1 -8@$2 -8@$3 -8@$4 -8@$5 -8@$6 -8@$7 -8@$8 -8@$9 -8@$10 -8@$11",
                 $arg1, $arg2, $arg3, $arg4, $arg5, $arg6, $arg7, $arg8, $arg9, $arg10, $arg11,
                 $arg12));
)

#[cfg(target_word_size = "32")]
#[macro_export]
macro_rules! sdt_asm(
    ($provider:ident, $name:ident, $argstr:tt $(, $arg:expr)*)
    => (unsafe {
        _sdt_asm!(".4byte", $provider, $name, $argstr $(, $arg)*);
    }))

#[cfg(target_word_size = "64")]
#[macro_export]
macro_rules! sdt_asm(
    ($provider:ident, $name:ident, $argstr:tt $(, $arg:expr)*)
    => (unsafe {
        _sdt_asm!(".8byte", $provider, $name, $argstr $(, $arg)*);
    }))

// Since we can't #include <sys/sdt.h>, we have to reinvent it...
// but once you take out the C/C++ type handling, there's not a lot to it.
#[macro_export]
macro_rules! _sdt_asm(
    ($addr:tt, $provider:ident, $name:ident, $argstr:tt $(, $arg:expr)*) => (
        asm!(concat!(r#"
990:    nop
        .pushsection .note.stapsdt,"?","note"
        .balign 4
        .4byte 992f-991f, 994f-993f, 3
991:    .asciz "stapsdt"
992:    .balign 4
993:    "#, $addr, r#" 990b
        "#, $addr, r#" _.stapsdt.base
        "#, $addr, r#" 0 // FIXME set semaphore address
        .asciz ""#, stringify!($provider), r#""
        .asciz ""#, stringify!($name), r#""
        .asciz ""#, $argstr, r#""
994:    .balign 4
        .popsection
.ifndef _.stapsdt.base
        .pushsection .stapsdt.base,"aG","progbits",.stapsdt.base,comdat
        .weak _.stapsdt.base
        .hidden _.stapsdt.base
_.stapsdt.base: .space 1
        .size _.stapsdt.base, 1
        .popsection
.endif
"#
            )
            : // output operands
            : // input operands
                $("nor"(($arg) as i64)),*
            : // clobbers
            : // options
                "volatile"
        )
    ))
