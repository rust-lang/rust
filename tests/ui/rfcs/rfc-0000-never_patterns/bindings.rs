#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {
    let x: Result<bool, &(u32, u32, Void)> = Ok(false);

    // FIXME(never_patterns): Never patterns in or-patterns don't need to share the same bindings.
    match x {
        Ok(_x) | Err(&!) => {}
        //~^ ERROR: is not bound in all patterns
    }
    let (Ok(_x) | Err(&!)) = x;
    //~^ ERROR: is not bound in all patterns

    // FIXME(never_patterns): A never pattern mustn't have bindings.
    match x {
        Ok(_) => {}
        Err(&(_a, _b, !)),
    }
    match x {
        Ok(_ok) | Err(&(_a, _b, !)) => {}
        //~^ ERROR: is not bound in all patterns
        //~| ERROR: is not bound in all patterns
        //~| ERROR: is not bound in all patterns
    }
}
