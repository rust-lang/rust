//@ check-pass
//@ known-bug: #84591

trait Subtrait<'a, 'b, R>: Supertrait<'a, 'b> {}
trait Supertrait<'a, 'b> {
    fn convert<T: ?Sized>(x: &'a T) -> &'b T;
}

fn need_hrtb_subtrait<'a_, 'b_, S, T: ?Sized>(x: &'a_ T) -> &'b_ T
where
    S: for<'a, 'b> Subtrait<'a, 'b, &'b &'a ()>,
{
    need_hrtb_supertrait::<S, T>(x)
    // This call works and drops the implied bound `'a: 'b`
    // of the where-bound. This means the where-bound can
    // now be used to transmute any two lifetimes.
}

fn need_hrtb_supertrait<'a_, 'b_, S, T: ?Sized>(x: &'a_ T) -> &'b_ T
where
    S: for<'a, 'b> Supertrait<'a, 'b>,
{
    S::convert(x)
}

struct MyStruct;
impl<'a: 'b, 'b> Supertrait<'a, 'b> for MyStruct {
    fn convert<T: ?Sized>(x: &'a T) -> &'b T {
        x
    }
}
impl<'a, 'b> Subtrait<'a, 'b, &'b &'a ()> for MyStruct {}

fn extend_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    need_hrtb_subtrait::<MyStruct, T>(x)
}

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = extend_lifetime(&x);
    }
    println!("{}", d);
}
