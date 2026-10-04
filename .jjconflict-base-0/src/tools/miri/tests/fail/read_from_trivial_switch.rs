// Ensure that we don't optimize out `SwitchInt` reads even if that terminator
// branches to the same basic block on every target, since the operand may have
// side-effects that affect analysis of the MIR.
//
// See <https://github.com/rust-lang/miri/issues/4237>.

//@ normalize-stderr-test: "(\n)ALLOC \(.*\) \{\n(.*\n)*\}(\n)" -> "${1}ALLOC DUMP${3}"
//@ normalize-stderr-test: "\[0x[0-9a-z]..0x[0-9a-z]\]" -> "[0xX..0xY]"

use std::mem::MaybeUninit;

fn main() {
    let uninit: MaybeUninit<i32> = MaybeUninit::uninit();
    let bad_ref: &i32 = unsafe { uninit.assume_init_ref() };
    let &(0 | _) = bad_ref;
    //~^ ERROR: /Undefined Behavior: .*, but memory is uninitialized .* requires initialized memory/
}
