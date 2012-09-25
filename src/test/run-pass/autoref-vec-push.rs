trait VecPush<T> {
    fn push(&mut self, +t: T);
}

impl<T> ~[T]: VecPush<T> {
    fn push(&mut self, +t: T) {
        vec::push(*self, t);
    }
}

fn main() {
    let mut x = ~[];
    x.push(1);
    x.push(2);
    x.push(3);
    assert x == ~[1, 2, 3];
}