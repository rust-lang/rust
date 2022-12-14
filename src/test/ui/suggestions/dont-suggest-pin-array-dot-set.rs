// https://github.com/rust-lang/rust/issues/96834
//
// This test case verifies that rustc does not make an unhelpful suggestion:
//
//     help: consider wrapping the receiver expression with the appropriate type
//         |
//     14  |     Pin::new(&mut a).set(0, 3);
//         |     +++++++++++++  +
//
// We can tell that it isn't helpful, because `Pin::set` takes two parameters (including
// the receiver), but the function call on line 14 supplies three.
fn main() {
    let mut a = [0u8; 1];
    a.set(0, 3); //~ERROR
}
