pub trait Foo {
    #[inline(always)]
    fn f(&self);
}

pub struct Bar {
    x: ~str
}

impl Bar : Foo {
    #[inline(always)]
    fn f(&self) {
        io::println((*self).x);
    }
}


