// compile-flags: -Zdrop-tracking-mir --edition=2021

#![feature(generators)]

pub async fn async_bad_body() {
    match true {} //~ ERROR non-exhaustive patterns: type `bool` is non-empty
}

pub fn generator_bad_body() {
    || {
        // 'non-exhaustive pattern' only seems to be reported once, so this annotation doesn't work
        // keep the function around so we can make sure it doesn't ICE
        match true {}; // ERROR non-exhaustive patterns: type `bool` is non-empty
        yield ();
    };
}

fn main() {}
