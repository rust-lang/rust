// check-pass
// compile-flags: -Z span-debug
// aux-build:test-macros.rs

// Regression test for issue #75930
// Tests that we cfg-strip all targets before invoking
// a derive macro

#[macro_use]
extern crate test_macros;

#[derive(Print)]
struct Foo<#[cfg(FALSE)] A, B> {
    #[cfg(FALSE)] first: String,
    second: bool,
    third: [u8; {
        #[cfg(FALSE)] struct Bar;
        #[cfg(not(FALSE))] struct Inner;
        #[cfg(FALSE)] let a = 25;
        match true {
            #[cfg(FALSE)] true => {},
            false => {},
            _ => {}
        };
        0
    }],
    fourth: B
}

fn main() {}
