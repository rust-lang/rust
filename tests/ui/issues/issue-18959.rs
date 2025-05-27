pub trait Foo { fn foo<T>(&self, ext_thing: &T); }
pub trait Bar: Foo { }
impl<T: Foo> Bar for T { }

pub struct Thing;
impl Foo for Thing {
    fn foo<T>(&self, _: &T) {}
}

#[inline(never)]
fn foo(b: &dyn Bar) {
    //~^ ERROR E0038
    b.foo(&0)
}

fn main() {
    let mut thing = Thing;
    let test: &dyn Bar = &mut thing;
    //~^ ERROR E0038
    foo(test);
}
