//! Regression test for <https://github.com/rust-lang/rust/issues/23958>.
//! Test method which returns associated type defined on trait with HRTB doesn't ICE.
//@ run-pass

trait Collection where for<'a> &'a Self: IntoIterator {
    fn my_iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T> Collection for [T] { }

fn main() {
    let v = [0usize];
    let _ = v.my_iter();
}
