type Foo = impl Send;
//~^ ERROR `impl Trait` in type aliases is unstable

struct A;

#[define_opaque(Foo)]
//~^ ERROR unstable library feature
const fn foo() -> Foo {
    value()
    //~^ ERROR cannot find function `value` in this scope
}

const VALUE: Foo = foo();

fn test() {
    match VALUE {
        0 | 0 => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        _ => (),
    }
}

fn main() {}
