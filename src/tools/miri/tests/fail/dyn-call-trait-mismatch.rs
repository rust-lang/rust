// Validation and SB stop this too early.
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows

trait T1 {
    #[allow(dead_code)]
    fn method1(self: Box<Self>);
}
trait T2 {
    fn method2(self: Box<Self>);
}

impl T1 for i32 {
    fn method1(self: Box<Self>) {}
}

fn main() {
    let r = Box::new(0) as Box<dyn T1>;
    let r2: Box<dyn T2> = unsafe { std::mem::transmute(r) };
    r2.method2(); //~ERROR: using vtable for `T1` but `T2` was expected
}
