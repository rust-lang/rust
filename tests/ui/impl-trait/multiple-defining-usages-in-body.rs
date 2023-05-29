trait Trait {}
impl Trait for () {}

fn foo<T: Trait, U: Trait>() -> impl Trait {
    //~^ WARN function cannot return without recursing [unconditional_recursion]
    let a: T = foo::<T, U>();
    //~^ ERROR concrete type differs from previous defining opaque type use
    loop {}
    let _: T = foo::<U, T>();
}

fn main() {}
