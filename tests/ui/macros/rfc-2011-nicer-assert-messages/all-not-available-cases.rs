// aux-build:common.rs
// ignore-tidy-linelength
// only-x86_64
// run-pass
// needs-unwind Asserting on contents of error message

#![feature(core_intrinsics, generic_assert)]

extern crate common;

#[derive(Clone, Copy, PartialEq)]
struct CopyNoDebug(i32);

#[derive(Debug, PartialEq)]
struct NoCopyDebug(i32);

#[derive(PartialEq)]
struct NoCopyNoDebug(i32);

fn main() {
  // Has Copy but does not have Debug
  common::test!(
    let mut copy_no_debug = CopyNoDebug(1);
    [ copy_no_debug == CopyNoDebug(3) ] => "Assertion failed: copy_no_debug == CopyNoDebug(3)\nWith captures:\n  copy_no_debug = N/A\n"
  );

  // Does not have Copy but has Debug
  common::test!(
    let mut no_copy_debug = NoCopyDebug(1);
    [ no_copy_debug == NoCopyDebug(3) ] => "Assertion failed: no_copy_debug == NoCopyDebug(3)\nWith captures:\n  no_copy_debug = N/A\n"
  );

  // Does not have Copy and does not have Debug
  common::test!(
    let mut no_copy_no_debug = NoCopyNoDebug(1);
    [ no_copy_no_debug == NoCopyNoDebug(3) ] => "Assertion failed: no_copy_no_debug == NoCopyNoDebug(3)\nWith captures:\n  no_copy_no_debug = N/A\n"
  );

  // Unevaluated (Expression short-circuited)
  common::test!(
    let mut elem = true;
    [ false && elem ] => "Assertion failed: false && elem\nWith captures:\n  elem = N/A\n"
  );
}
