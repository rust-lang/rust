pub trait Foo {
    #[inline(always)]
    fn f(&self);
}

pub struct Bar {
    pub x: String
}

impl Foo for Bar {
    #[inline(always)]
    fn f(&self) {
        println!("{}", (*self).x);
    }
}
