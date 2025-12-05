//@ test-mir-pass: ElaborateDrops
//@ needs-unwind

#![feature(rustc_attrs, liballoc_internals)]

// EMIT_MIR box_partial_move.maybe_move.ElaborateDrops.diff
fn maybe_move(cond: bool, thing: Box<String>) -> Option<String> {
    // CHECK-LABEL: fn maybe_move(
    // CHECK: let mut [[PTR:_[0-9]+]]: *const std::string::String;
    // CHECK: [[PTR]] = copy ((_2.0: std::ptr::Unique<std::string::String>).0: std::ptr::NonNull<std::string::String>) as *const std::string::String (Transmute);
    // CHECK: drop((*[[PTR]]))
    if cond { Some(*thing) } else { None }
}

fn main() {
    maybe_move(false, Box::new("hello".to_string()));
}
