// compile-flags: -Zverbose

#![feature(generators, generator_trait)]

fn main() {
    let x = "Type mismatch test";
    let generator :() = || {
    //~^ ERROR mismatched types
        yield 1i32;
        return x
    };
}
