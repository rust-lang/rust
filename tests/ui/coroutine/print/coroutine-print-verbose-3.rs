// compile-flags: -Zverbose

#![feature(coroutines, coroutine_trait)]

fn main() {
    let x = "Type mismatch test";
    let coroutine :() = || {
    //~^ ERROR mismatched types
        yield 1i32;
        return x
    };
}
