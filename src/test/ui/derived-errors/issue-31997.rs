// Test that the resolve failure does not lead to downstream type errors.
// See issue #31997.

trait TheTrait { }

fn closure<F, T>(x: F) -> Result<T, ()>
    where F: FnMut() -> T, T: TheTrait,
{
    unimplemented!()
}

fn foo() -> Result<(), ()> {
    try!(closure(|| bar(core::ptr::null_mut()))); //~ ERROR cannot find function `bar` in this scope
    Ok(())
}

fn main() { }
