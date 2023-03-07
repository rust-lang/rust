// run-pass


trait Pushable<T> {
    fn push_val(&mut self, t: T);
}

impl<T> Pushable<T> for Vec<T> {
    fn push_val(&mut self, t: T) {
        self.push(t);
    }
}

pub fn main() {
    let mut v = vec![1];
    v.push_val(2);
    v.push_val(3);
    assert_eq!(v, [1, 2, 3]);
}
