trait Pushable<T> {
    fn push_val(&mut self, +t: T);
}

impl<T> ~[T]: Pushable<T> {
    fn push_val(&mut self, +t: T) {
        self.push(t);
    }
}

fn main() {
    let mut v = ~[1];
    v.push_val(2);
    v.push_val(3);
    assert v == ~[1, 2, 3];
}