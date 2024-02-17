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
