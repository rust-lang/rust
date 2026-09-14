//! Regression test for <https://github.com/rust-lang/rust/issues/26709>.
//! Test value is still getting dropped after unsized coerce.
//!
//! Fat pointer was passed in two immediate args, but the drop invocation
//! used to accept one, which led to ICE.
//@ run-pass

struct Wrapper<'a, T: ?Sized>(&'a mut i32, #[allow(dead_code)] T);

impl<'a, T: ?Sized> Drop for Wrapper<'a, T> {
    fn drop(&mut self) {
        *self.0 = 432;
    }
}

fn main() {
    let mut x = 0;
    {
        let wrapper = Box::new(Wrapper(&mut x, 123));
        let _: Box<Wrapper<dyn Send>> = wrapper;
    }
    assert_eq!(432, x)
}
