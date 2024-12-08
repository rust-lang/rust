// Check that we don't have a cycle when we try to normalize `Self::U` in the
// bound below. Make sure that having a lifetime on the trait object doesn't break things

//@ check-pass

trait Is {
    type T;
}

impl<U> Is for U {
    type T = U;
}

trait Obj<'a> {
    type U: Is<T = Self::V>;
    type V;
}

fn is_obj<'a, T: ?Sized + Obj<'a>>(_: &T) {}

fn f<'a>(x: &dyn Obj<'a, U = i32, V = i32>) {
    is_obj(x)
}

fn main() {}
