trait Foo { const FOO: Self; }
impl Foo for u32 { const FOO: Self = 1; }
fn bar<T: Foo>(n: T) {
    const BASE: T = T::FOO;
    //~^ ERROR can't use generic parameters in `const` associated item
    //~| ERROR can't use generic parameters in `const` associated item
    type Type = T;
    //~^ ERROR can't use generic parameters in associated type
}
fn main() {}
