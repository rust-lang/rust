#![feature(never_type)]
#![feature(exhaustive_patterns)]
#![deny(unreachable_patterns)]
//~^ NOTE lint level is defined here

#[rustfmt::skip]
fn main() {
    match (0u8,) {
        (1 | 2,) => {}
        //~^ NOTE matches all the relevant values
        (2,) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
        _ => {}
    }

    match (0u8,) {
        (1,) => {}
        //~^ NOTE matches some of the same values
        (2,) => {}
        //~^ NOTE matches some of the same values
        (1 | 2,) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
        //~| NOTE multiple earlier patterns match some of the same values
        //~| NOTE collectively making this unreachable
        _ => {}
    }

    match 0u8 {
        1 => {}
        //~^ NOTE matches some of the same values
        2 => {}
        //~^ NOTE matches some of the same values
        3 => {}
        //~^ NOTE matches some of the same values
        4 => {}
        //~^ NOTE matches some of the same values
        5 => {}
        6 => {}
        1 ..= 6 => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
        //~| NOTE multiple earlier patterns match some of the same values
        //~| NOTE ...and 2 other patterns
        _ => {}
    }

    let res: Result<(),!> = Ok(());
    match res {
        Ok(_) => {}
        Err(_) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE matches no values because `!` is uninhabited
        //~| NOTE to learn more about uninhabited types, see
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
        //~| NOTE matches no values because `Void2` is uninhabited
        //~| NOTE to learn more about uninhabited types, see
        _ => {}
    }
    match (res1, &res2) {
        (Err(_), Err(_)) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE matches no values because `Void1` is uninhabited
        //~| NOTE to learn more about uninhabited types, see
        _ => {}
    }


    if let (0
        //~^ NOTE matches all the relevant values
        | 0, _) = (0, 0) {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this

    match (true, true) {
        (_, true) if false => {} // Guarded patterns don't cover others
        (true, _) => {}
        //~^ NOTE matches some of the same values
        (false, _) => {}
        //~^ NOTE matches some of the same values
        (_, true) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
        //~| NOTE multiple earlier patterns match some of the same values
        //~| NOTE collectively making this unreachable
    }

    match (true, true) {
        (true, _) => {}
        //~^ NOTE matches all the relevant values
        (false, _) => {}
        #[allow(unreachable_patterns)]
        (_, true) => {} // Doesn't cover below because it's already unreachable.
        (true, true) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
    }

    // Despite skipping some irrelevant cases, we still report a set of rows that covers the
    // unreachable one.
    match (true, true, 0) {
        (true, _, _) => {}
        (_, true, 0..10) => {}
        //~^ NOTE matches all the relevant values
        (_, true, 10..) => {}
        (_, true, 3) => {}
        //~^ ERROR unreachable pattern
        //~| NOTE no value can reach this
        _ => {}
    }
}
