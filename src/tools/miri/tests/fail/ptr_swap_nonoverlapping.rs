//! This is a regression test for <https://github.com/rust-lang/miri/issues/4188>: The precondition
//! check in `ptr::swap_nonoverlapping` was incorrectly disabled in Miri.
//@normalize-stderr-test: "\|.*::abort\(\).*" -> "| ABORT()"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
//@error-in-other-file: aborted execution

fn main() {
    let mut data = 0usize;
    let ptr = std::ptr::addr_of_mut!(data);
    unsafe {
        std::ptr::swap_nonoverlapping(ptr, ptr, 1);
    }
}
