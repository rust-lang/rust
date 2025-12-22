//@ check-pass
// This checks that the compiler does not require that 'a: 'b. '_ has 'a and 'b as non-local
// upper bounds, but the compiler should not propagate 'a: 'b OR 'b: 'a when checking
// the closures. If it did, this would fail to compile, eventhough it's a valid program.
// PR #148329 explains this in detail.

struct MyTy<'x, 'a, 'b>(std::cell::Cell<(&'x &'a u8, &'x &'b u8)>);
fn wf<T>(_: T) {}
fn test<'a, 'b>() {
    |_: &'a u8, x: MyTy<'_, 'a, 'b>| wf(x);
    |x: MyTy<'_, 'a, 'b>, _: &'a u8| wf(x);
}

fn main(){}
