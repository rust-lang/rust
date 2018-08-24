// Testing that the B's are resolved


trait clam<A> { fn get(self) -> A; }

struct foo(isize);

impl foo {
    pub fn bar<B,C:clam<B>>(&self, _c: C) -> B { panic!(); }
}

pub fn main() { }
