// issue-119209

type Foo<'a> = impl PartialEq; //~ERROR `impl Trait` in type aliases is unstable

fn main<'a>(_: &'a i32) -> Foo<'a> {} //~ERROR `main` function return type is not allowed to have generic parameters
