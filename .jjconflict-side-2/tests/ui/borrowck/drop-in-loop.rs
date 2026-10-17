// A version of `issue-70919-drop-in-loop`, but without
// the necessary `drop` call.
//
// This should fail to compile, since the `Drop` impl
// for `WrapperWithDrop` could observe the changed
// `base` value.

struct WrapperWithDrop<'a>(&'a mut bool);
impl<'a> Drop for WrapperWithDrop<'a> {
    fn drop(&mut self) {
    }
}

fn drop_in_loop() {
    let mut base = true;
    let mut wrapper = WrapperWithDrop(&mut base);
    loop {
        base = false; //~ ERROR: cannot assign to `base`
        wrapper = WrapperWithDrop(&mut base);
    }
}

fn main() {
}
