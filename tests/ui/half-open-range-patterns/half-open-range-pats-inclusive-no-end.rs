// Test `X...` and `X..=` range patterns not being allowed syntactically.
// FIXME(Centril): perhaps these should be semantic restrictions.

fn main() {}

#[cfg(false)]
fn foo() {
    if let 0... = 1 {} //~ ERROR inclusive range with no end
    if let 0..= = 1 {} //~ ERROR inclusive range with no end
    const X: u8 = 0;
    if let X... = 1 {} //~ ERROR inclusive range with no end
    if let X..= = 1 {} //~ ERROR inclusive range with no end
}

fn bar() {
    macro_rules! mac {
        ($e:expr) => {
            let $e...; //~ ERROR inclusive range with no end
            //~^ ERROR: refutable pattern
            let $e..=; //~ ERROR inclusive range with no end
            //~^ ERROR: refutable pattern
        }
    }

    mac!(0);
}
