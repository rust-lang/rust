// Case that the fix for #74868 also allowed to compile

//@ check-pass

trait BoxedDsl {
    type Output;
}

impl<T> BoxedDsl for T
where
    T: BoxedDsl,
{
    type Output = <T as BoxedDsl>::Output;
}

trait HandleUpdate {}

impl<T> HandleUpdate for T where T: BoxedDsl<Output = ()> {}

fn main() {}
