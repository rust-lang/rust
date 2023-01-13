// run-pass
// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.


trait Get<T> {
    fn get(&self) -> T;
}

#[derive(Copy, Clone)]
struct Box<T> {
    t: T
}

impl<T:Clone> Get<T> for Box<T> {
    fn get(&self) -> T {
        self.t.clone()
    }
}

fn add<'a,G:Get<&'a isize>>(g1: G, g2: G) -> isize {
    *g1.get() + *g2.get()
}

pub fn main() {
    let b1 = Box { t: &3 };
    assert_eq!(add(b1, b1), 6);
}
