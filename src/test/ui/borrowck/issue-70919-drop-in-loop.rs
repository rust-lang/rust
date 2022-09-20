// Regression test for issue #70919
// Tests that we don't emit a spurious "borrow might be used" error
// when we have an explicit `drop` in a loop

// check-pass

struct WrapperWithDrop<'a>(&'a mut bool);
impl<'a> Drop for WrapperWithDrop<'a> {
    fn drop(&mut self) {
    }
}

fn drop_in_loop() {
    let mut base = true;
    let mut wrapper = WrapperWithDrop(&mut base);
    loop {
        drop(wrapper);

        base = false;
        wrapper = WrapperWithDrop(&mut base);
    }
}

fn main() {
}
