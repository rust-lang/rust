//@ check-pass


struct MyTy<'x, 'a, 'b>(std::cell::Cell<(&'x &'a u8, &'x &'b u8)>);
fn wf<T>(_: T) {}
fn test<'a, 'b>() {
    |_: &'a u8, x: MyTy<'_, 'a, 'b>| wf(x);
    |x: MyTy<'_, 'a, 'b>, _: &'a u8| wf(x);
}

fn main(){}
