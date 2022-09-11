type Foo<'a> = &'a dyn Send + Sync;
type Foo = *const dyn Send + Sync;
type Foo = fn() -> dyn Send + 'static;
fn main() {
    let b = (&a) as &dyn Add<Other, Output = Addable> + Other;
}
