type Foo = impl Send;
//~^ ERROR `impl Trait` in type aliases is unstable

struct A;

const VALUE: Foo = value();
//~^ ERROR cannot find function `value` in this scope

fn test() {
    match VALUE {
        0 | 0 => {}
//~^ ERROR mismatched types
//~| ERROR mismatched types
        _ => (),
    }
}

fn main() {}
