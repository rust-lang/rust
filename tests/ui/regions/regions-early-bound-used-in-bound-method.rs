//@ run-pass
// Tests that you can use a fn lifetime parameter as part of
// the value for a type parameter in a bound.


trait GetRef<'a> {
    fn get(&self) -> &'a isize;
}

#[derive(Copy, Clone)]
struct Box<'a> {
    t: &'a isize
}

impl<'a> GetRef<'a> for Box<'a> {
    fn get(&self) -> &'a isize {
        self.t
    }
}

impl<'a> Box<'a> {
    fn add<'b,G:GetRef<'b>>(&self, g2: G) -> isize {
        *self.t + *g2.get()
    }
}

pub fn main() {
    let b1 = Box { t: &3 };
    assert_eq!(b1.add(b1), 6);
}
