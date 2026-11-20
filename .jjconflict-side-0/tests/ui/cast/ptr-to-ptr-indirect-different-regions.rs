// We want to forbid extending lifetimes on object types behind ptrs
// as it may allow for uncallable VTable methods to become accessible.

trait Trait {
    fn foo(&self) {}
}

struct MyWrap<T: ?Sized>(T);

fn bar<'a>(a: *mut MyWrap<(dyn Trait + 'a)>) -> *mut MyWrap<(dyn Trait + 'static)> {
    a as _
    //~^ ERROR: lifetime may not live long enough
}

fn main() {}
