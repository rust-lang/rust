// Check that we don't have a cycle when we try to normalize `Self::U` in the
// bound below.

//@ check-pass

trait Is {
    type T;
}

impl<U> Is for U {
    type T = U;
}

trait Obj {
    type U: Is<T = Self::U>;
}

fn is_obj<T: ?Sized + Obj>(_: &T) {}

fn f(x: &dyn Obj<U = i32>) {
    is_obj(x)
}

fn main() {}
