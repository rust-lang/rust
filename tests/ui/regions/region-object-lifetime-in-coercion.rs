// Test that attempts to implicitly coerce a value into an
// object respect the lifetime bound on the object type.

trait Foo {}
impl<'a> Foo for &'a [u8] {}

fn a(v: &[u8]) -> Box<dyn Foo + 'static> {
    let x: Box<dyn Foo + 'static> = Box::new(v);
    //~^ ERROR lifetime may not live long enough
    x
}

fn b(v: &[u8]) -> Box<dyn Foo + 'static> {
    Box::new(v)
    //~^ ERROR lifetime may not live long enough
}

fn c(v: &[u8]) -> Box<dyn Foo> {
    // same as previous case due to RFC 599

    Box::new(v)
    //~^ ERROR lifetime may not live long enough
}

fn d<'a,'b>(v: &'a [u8]) -> Box<dyn Foo+'b> {
    Box::new(v)
    //~^ ERROR lifetime may not live long enough
}

fn e<'a:'b,'b>(v: &'a [u8]) -> Box<dyn Foo+'b> {
    Box::new(v) // OK, thanks to 'a:'b
}

fn main() { }
