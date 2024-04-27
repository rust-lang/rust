// Check that safe fns are not a subtype of unsafe fns.

trait Foo {
    unsafe fn len(&self) -> u32;
}

impl Foo for u32 {
    fn len(&self) -> u32 { *self }
    //~^ ERROR method `len` has an incompatible type for trait
    //~| expected signature `unsafe fn(&_) -> _`
    //~| found signature `fn(&_) -> _`
}

fn main() { }
