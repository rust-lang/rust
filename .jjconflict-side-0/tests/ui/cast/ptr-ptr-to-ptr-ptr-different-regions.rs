//@ check-pass

// We allow extending lifetimes of object types if they are behind two layers
// of pointer indirection (as opposed to one). This is because this is the more
// general case of casting between two sized types (`*mut T as *mut U`).

trait Trait {
    fn foo(&self) {}
}

fn bar<'a>(a: *mut *mut (dyn Trait + 'a)) -> *mut *mut (dyn Trait + 'static) {
    a as _
}

fn main() {}
