// output doesn't get corrupted when using comments within generic type parameters of a trait

pub trait Something<
    A,
    // some comment
    B,
    C
> {
    fn a(&self, x: A) -> i32;
    fn b(&self, x: B) -> i32;
    fn c(&self, x: C) -> i32;
}

pub trait SomethingElse<
    A,
    /* some comment */
    B,
    C
> {
    fn a(&self, x: A) -> i32;
    fn b(&self, x: B) -> i32;
    fn c(&self, x: C) -> i32;
}
