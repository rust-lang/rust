// Checks that range metadata gets emitted on calls to functions returning a
// scalar value.

// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

pub fn test() {
    // CHECK: call noundef i8 @some_true(){{( #[0-9]+)?}}, !range [[R0:![0-9]+]]
    // CHECK: [[R0]] = !{i8 0, i8 3}
    some_true();
}

#[no_mangle]
fn some_true() -> Option<bool> {
    Some(true)
}
