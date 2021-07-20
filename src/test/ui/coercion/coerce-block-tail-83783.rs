// build-pass
// edition:2018
fn _consume_reference<T: ?Sized>(_: &T) {}

async fn _foo() {
    _consume_reference::<i32>(&Box::new(7_i32)); // OK.
    _consume_reference::<i32>(&async { Box::new(7_i32) }.await); // Error.
    _consume_reference::<[i32]>(&vec![7_i32]); // OK.
    _consume_reference::<[i32]>(&async { vec![7_i32] }.await); // OK.
}

fn main() { }
