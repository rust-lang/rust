#![feature(never_patterns)]
#![allow(incomplete_features)]

#[derive(Copy, Clone)]
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

fn void(void: Void) {
    let (_a | !) = void;
    let (! | _a) = void;
    let ((_a, _) | (_a, _ | !)) = (true, void);
    let (_a | (! | !,)) = (void,);
    let ((_a,) | (!,)) = (void,);

    let (_a, (! | !)) = (true, void);
    //~^ ERROR: never patterns cannot contain variable bindings
    let (_a, (_b | !)) = (true, void);

    let _a @ ! = void;
    //~^ ERROR: never patterns cannot contain variable bindings
    let _a @ (_b | !) = void;
    let (_a @ (), !) = ((), void);
    //~^ ERROR: never patterns cannot contain variable bindings
    let (_a |
            (_b @ (_, !))) = (true, void);
    //~^ ERROR: never patterns cannot contain variable bindings
}
