//@ check-pass
// FIXME: add explanation.

struct MyTy<'a, 'b, 'x>(std::cell::Cell<(&'a &'x str, &'b &'x str)>);
fn wf<T>(_: T) {}
fn test<'a, 'b, 'x>() {
    |x: MyTy<'a, 'b, '_>| wf(x);
}

fn main() {}
