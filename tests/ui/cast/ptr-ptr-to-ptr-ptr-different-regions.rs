//@ check-pass

trait Trait {
    fn foo(&self) {}
}

fn bar<'a>(a: *mut *mut (dyn Trait + 'a)) -> *mut *mut (dyn Trait + 'static) {
    a as _
}

fn main() {}
