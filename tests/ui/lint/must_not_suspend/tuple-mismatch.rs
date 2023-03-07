#![feature(generators)]

fn main() {
    let _generator = || {
        yield ((), ((), ()));
        yield ((), ());
        //~^ ERROR mismatched types
    };
}
