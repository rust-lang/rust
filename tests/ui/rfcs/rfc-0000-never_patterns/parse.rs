#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn main() {}

macro_rules! never {
    () => { ! }
}

fn parse(x: Void) {
    match None::<Void> {
        None => {}
        Some(!),
    }
    match None::<Void> {
        Some(!),
        None => {}
    }
    match None::<Void> {
        None => {}
        Some(!)
    }
    match None::<Void> {
        Some(!)
        //~^ ERROR expected `,` following `match` arm
        None => {}
    }
    match None::<Void> {
        Some(!) if true
        //~^ ERROR expected `,` following `match` arm
        //~| ERROR guard on a never pattern
        None => {}
    }
    match None::<Void> {
        Some(!) if true,
        //~^ ERROR guard on a never pattern
        None => {}
    }
    match None::<Void> {
        Some(!) <=
        //~^ ERROR expected one of
    }
    match x {
        never!(),
    }
    match x {
        never!() if true,
        //~^ ERROR guard on a never pattern
    }
    match x {
        never!()
    }
    match &x {
        &never!(),
    }
    match None::<Void> {
        Some(never!()),
        None => {}
    }
    match x { ! }
    match &x { &! }

    let res: Result<bool, Void> = Ok(false);
    let Ok(_) = res;
    let Ok(_) | Err(!) = &res; // Disallowed; see #82048.
    //~^ ERROR `let` bindings require top-level or-patterns in parentheses
    let (Ok(_) | Err(!)) = &res;
    let (Ok(_) | Err(&!)) = res.as_ref();

    let ! = x;
    let y @ ! = x;
    //~^ ERROR: never patterns cannot contain variable bindings
}

fn foo(!: Void) {}
