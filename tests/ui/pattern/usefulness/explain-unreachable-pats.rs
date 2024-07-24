#![feature(never_type)]
#![feature(min_exhaustive_patterns)]
#![deny(unreachable_patterns)]
//~^ NOTE lint level is defined here

#[rustfmt::skip]
fn main() {
    match (0u8,) {
        (1 | 2,) => {}
        //~^ NOTE matches all the values already
        (2,) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern
        _ => {}
    }

    match (0u8,) {
        (1,) => {}
        //~^ NOTE matches some of the same values
        (2,) => {}
        //~^ NOTE matches some of the same values
        (1 | 2,) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern
        //~| NOTE these patterns collectively make the last one unreachable
        //~| NOTE collectively making this unreachable
        _ => {}
    }

    let res: Result<(),!> = Ok(());
    match res {
        Ok(_) => {}
        Err(_) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE this pattern matches no values because `!` is uninhabited
    }

    #[derive(Copy, Clone)]
    enum Void1 {}
    #[derive(Copy, Clone)]
    enum Void2 {}
    // Only an empty type matched _by value_ can make an arm unreachable. We must get the right one.
    let res1: Result<(), Void1> = Ok(());
    let res2: Result<(), Void2> = Ok(());
    match (&res1, res2) {
        (Err(_), Err(_)) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE this pattern matches no values because `Void2` is uninhabited
        _ => {}
    }
    match (res1, &res2) {
        (Err(_), Err(_)) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE this pattern matches no values because `Void1` is uninhabited
        _ => {}
    }


    if let (0
        //~^ NOTE matches all the values already
        | 0, _) = (0, 0) {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern

    match (true, true) {
        (_, true) if false => {} // Guarded patterns don't cover others
        (true, _) => {}
        //~^ NOTE matches some of the same values
        (false, _) => {}
        //~^ NOTE matches some of the same values
        (_, true) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern
        //~| NOTE these patterns collectively make the last one unreachable
        //~| NOTE collectively making this unreachable
    }

    match (true, true) {
        (true, _) => {}
        //~^ NOTE matches all the values already
        (false, _) => {}
        #[allow(unreachable_patterns)]
        (_, true) => {} // Doesn't cover below because it's already unreachable.
        (true, true) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern
    }

    // Despite skipping some irrelevant cases, we still report a set of rows that covers the
    // unreachable one.
    match (true, true, 0) {
        (true, _, _) => {}
        (_, true, 0..10) => {}
        //~^ NOTE matches all the values already
        (_, true, 10..) => {}
        (_, true, 3) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE unreachable pattern
        _ => {}
    }
}
