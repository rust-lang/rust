// Check that safe fns are not a subtype of unsafe fns.

//@ dont-require-annotations: NOTE

trait Foo {
    unsafe fn len(&self) -> u32;
}

impl Foo for u32 {
    fn len(&self) -> u32 { *self }
    //~^ ERROR method `len` has an incompatible type for trait
    //~| NOTE expected signature `unsafe fn(&_) -> _`
    //~| NOTE found signature `fn(&_) -> _`
}

fn main() { }
