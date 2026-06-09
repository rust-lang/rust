// Check that backtrace info is correctly generated for dynamic libraries and is usable by a
// rust binary.
// Part of porting some backtrace tests to rustc: <https://github.com/rust-lang/rust/issues/122899>.
// Original test:
// <https://github.com/rust-lang/backtrace-rs/tree/6fa4b85b9962c3e1be8c2e5cc605cd078134152b/crates/dylib-dep>
// ignore-tidy-linelength
//@ ignore-android FIXME #17520
//@ ignore-fuchsia Backtraces not symbolized
//@ ignore-musl musl doesn't support dynamic libraries (at least when the original test was written).
//@ ignore-ios needs the `.dSYM` files to be moved to the device
//@ ignore-tvos needs the `.dSYM` files to be moved to the device
//@ ignore-watchos needs the `.dSYM` files to be moved to the device
//@ ignore-visionos needs the `.dSYM` files to be moved to the device
//@ needs-unwind
//@ ignore-backends: gcc
//@ compile-flags: -g -Copt-level=0 -Cstrip=none -Cforce-frame-pointers=yes
//@ ignore-emscripten Requires custom symbolization code
//@ aux-crate: dylib_dep_helper=dylib-dep-helper.rs
//@ aux-crate: auxiliary=dylib-dep-helper-aux.rs
//@ run-pass

#![allow(improper_ctypes)]
#![allow(improper_ctypes_definitions)]

extern crate dylib_dep_helper;
extern crate auxiliary;

use std::backtrace::Backtrace;

macro_rules! pos {
    () => {
        (file!(), line!())
    };
}

#[collapse_debuginfo(yes)]
macro_rules! check {
    ($($pos:expr),*) => ({
        verify(&[$($pos,)* pos!()]);
    })
}

fn verify(filelines: &[Pos]) {
    let trace = Backtrace::capture();
    eprintln!("-----------------------------------");
    eprintln!("looking for:");
    for (file, line) in filelines.iter().rev() {
        eprintln!("\t{file}:{line}");
    }
    eprintln!("found:\n{trace:#?}");
    let mut iter = filelines.iter().rev();
    // FIXME(jieyouxu): use proper `BacktraceFrame` accessors when it becomes available. Right now,
    // this depends on the debug format of `Backtrace` which is of course fragile.
    let backtrace = format!("{:#?}", trace);
    while let Some((file, line)) = iter.next() {
        // FIXME(jieyouxu): make this test use proper accessors on `BacktraceFrames` once it has
        // them.
        assert!(backtrace.contains(file), "expected backtrace to contain {}", file);
        assert!(backtrace.contains(&line.to_string()), "expected backtrace to contain {}", line);
    }
}

type Pos = (&'static str, u32);

extern "C" {
    #[link_name = "foo"]
    fn foo(p: Pos, cb: fn(Pos, Pos));
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    unsafe {
        foo(pos!(), |a, b| {
            check!(a, b)
        })
    }

    outer(pos!());
}

#[inline(never)]
fn outer(main_pos: Pos) {
    inner(main_pos, pos!());
    inner_inlined(main_pos, pos!());
}

#[inline(never)]
fn inner(main_pos: Pos, outer_pos: Pos) {
    check!(main_pos, outer_pos);
    check!(main_pos, outer_pos);
    let inner_pos = pos!(); auxiliary::callback(|aux_pos| {
        check!(main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); auxiliary::callback_inlined(|aux_pos| {
        check!(main_pos, outer_pos, inner_pos, aux_pos);
    });
}

#[inline(always)]
fn inner_inlined(main_pos: Pos, outer_pos: Pos) {
    check!(main_pos, outer_pos);
    check!(main_pos, outer_pos);

    #[inline(always)]
    fn inner_further_inlined(main_pos: Pos, outer_pos: Pos, inner_pos: Pos) {
        check!(main_pos, outer_pos, inner_pos);
    }
    inner_further_inlined(main_pos, outer_pos, pos!());

    let inner_pos = pos!(); auxiliary::callback(|aux_pos| {
        check!(main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); auxiliary::callback_inlined(|aux_pos| {
        check!(main_pos, outer_pos, inner_pos, aux_pos);
    });

    // this tests a distinction between two independent calls to the inlined function.
    // (un)fortunately, LLVM somehow merges two consecutive such calls into one node.
    inner_further_inlined(main_pos, outer_pos, pos!());
}
