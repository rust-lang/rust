#![feature(coroutines)]

fn main() {
    let _coroutine = || {
        yield ((), ((), ()));
        yield ((), ());
        //~^ ERROR mismatched types
    };
}
