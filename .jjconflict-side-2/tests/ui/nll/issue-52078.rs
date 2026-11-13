// Regression test for #52078: we were failing to infer a relationship
// between `'a` and `'b` below due to inference variables introduced
// during the normalization process.
//
//@ check-pass

struct Drain<'a, T: 'a> {
    _marker: ::std::marker::PhantomData<&'a T>,
}

trait Join {
    type Value;
    fn get(value: &mut Self::Value);
}

impl<'a, T> Join for Drain<'a, T> {
    type Value = &'a mut Option<T>;

    fn get<'b>(value: &'b mut Self::Value) {
    }
}

fn main() {
}
