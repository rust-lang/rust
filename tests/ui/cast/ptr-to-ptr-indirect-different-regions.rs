trait Trait {
    fn foo(&self) {}
}

struct MyWrap<T: ?Sized>(T);

fn bar<'a>(a: *mut MyWrap<(dyn Trait + 'a)>) -> *mut MyWrap<(dyn Trait + 'static)> {
    a as _
    //~^ ERROR: lifetime may not live long enough
}

fn main() {}
