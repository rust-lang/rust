#![feature(type_alias_impl_trait)]

fn main() {
    type Closure = impl Fn() -> u64;
    struct Anonymous(Closure);
    let y = || -> Closure { || 3 };
    Anonymous(|| {
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        3
    })
}
