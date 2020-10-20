use std::collections::HashMap;

fn what() {
    let descr = String::new();
    let mut opts = HashMap::<String, ()>::new();
    let opt = String::new();

    opts.get(opt.as_ref()); //~ ERROR type annotations needed
}

fn main() {
    let ips: Vec<_> = (0..100_000).map(|_| u32::from(0u32.into())).collect();
    //~^ ERROR type annotations needed
}

trait Foo<'a, T: ?Sized> {
    fn foo(&self) -> Box<T> {
        todo!()
    }
}

trait Bar<'a, T: ?Sized> {
    fn bar(&self) -> Box<T> {
        todo!()
    }
}

impl Foo<'static, u32> for () {}
impl<'a> Foo<'a, i16> for () {}

impl<'a> Bar<'static, u32> for &'a () {}
impl<'a> Bar<'a, i16> for &'a () {}

fn foo() {
    let _ = ().foo(); //~ ERROR type annotations needed
}

fn bar() {
    let _ = (&()).bar(); //~ ERROR type annotations needed
}
