//! Ensure a decent error message for maybe-null references.
//! (see <https://github.com/rust-lang/rust/issues/146748>)

// Strip out raw byte dumps to make comparison platform-independent:
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*A(LLOC)?[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

#![feature(const_trait_impl, const_cmp)]

use std::any::TypeId;
use std::mem::transmute;

const A: [&(); 16 / size_of::<*const ()>()] = unsafe { transmute(TypeId::of::<i32>()) };
//~^ERROR: maybe-null

fn main() {}
