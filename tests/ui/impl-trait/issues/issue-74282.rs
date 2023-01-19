#![feature(type_alias_impl_trait)]

type Closure = impl Fn() -> u64;
struct Anonymous(Closure);

fn fop()
where
    Anonymous: 'static,
{
    let y = || -> Closure { || 3 };
    Anonymous(|| {
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        3
    })
}

fn main() {
    fop();
}
