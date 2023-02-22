// run-pass
#![allow(ref_binop_on_copy_type)]

fn main() {
    // We shouldn't promote this
    let _ = &(main as fn() == main as fn());
    // Also check nested case
    let _ = &(&(main as fn()) == &(main as fn()));
}
