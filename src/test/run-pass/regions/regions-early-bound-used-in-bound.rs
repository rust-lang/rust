// run-pass
// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.


trait GetRef<'a, T> {
    fn get(&self) -> &'a T;
}

#[derive(Copy, Clone)]
struct Box<'a, T:'a> {
    t: &'a T
}

impl<'a,T:Clone> GetRef<'a,T> for Box<'a,T> {
    fn get(&self) -> &'a T {
        self.t
    }
}

fn add<'a,G:GetRef<'a, isize>>(g1: G, g2: G) -> isize {
    *g1.get() + *g2.get()
}

pub fn main() {
    let b1 = Box { t: &3 };
    assert_eq!(add(b1, b1), 6);
}
