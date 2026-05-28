// Regression test for https://github.com/rust-lang/rust/issues/105677.
// This test ensures that the "Read more" link is only generated when
// there is actually more documentation to read after the short summary.

#![crate_name = "foo"]

pub trait MyFrom {
    /// # Hello
    /// ## Yolo
    /// more!
    fn try_from1();
    /// a
    /// b
    /// c
    fn try_from2();
    /// a
    ///
    /// b
    ///
    /// c
    fn try_from3();
}

pub struct NonZero;

//@ has 'foo/struct.NonZero.html'
impl MyFrom for NonZero {
    //@ matches - '//*[@class="docblock"]' '^Hello Read more$'
    fn try_from1() {}
    //@ matches - '//*[@class="docblock"]' '^a\sb\sc$'
    fn try_from2() {}
    //@ matches - '//*[@class="docblock"]' '^a Read more$'
    fn try_from3() {}
}
