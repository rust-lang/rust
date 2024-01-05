#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {
    let x: Result<bool, &(u32, u32, Void)> = Ok(false);

    match x {
        Ok(_x) | Err(&!) => {}
    }
    let (Ok(_x) | Err(&!)) = x;

    match x {
        Ok(_) => {}
        Err(&(_a, _b, !)),
        //~^ ERROR: never patterns cannot contain variable bindings
        //~| ERROR: never patterns cannot contain variable bindings
    }
    match x {
        Ok(_ok) | Err(&(_a, _b, !)) => {}
        //~^ ERROR: never patterns cannot contain variable bindings
        //~| ERROR: never patterns cannot contain variable bindings
    }
}
