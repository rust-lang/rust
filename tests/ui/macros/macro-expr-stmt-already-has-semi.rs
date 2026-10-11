// Regression test for issue #164087: do not suggest adding a semicolon
// after a macro invocation when one is already present.

macro_rules! a {
    () => {
        trait A {};
        0
    }; //~^ ERROR expected expression
}

fn main() {
    let _a = a!();
}
