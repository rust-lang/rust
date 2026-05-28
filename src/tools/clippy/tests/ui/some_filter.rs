#![warn(clippy::some_filter)]
#![allow(clippy::const_is_empty)]

macro_rules! unchanged {
    ($result:expr) => {
        $result
    };
}

macro_rules! condition {
    ($condition:expr) => {
        $condition || false
    };
}

#[clippy::msrv = "1.61"]
fn older() {
    let _ = Some(0).filter(|_| false);
}

#[clippy::msrv = "1.62"]
fn newer() {
    let _ = Some(0).filter(|_| false);
    //~^ some_filter
}

fn main() {
    let _ = Some(0).filter(|_| false);
    //~^ some_filter

    // The condition contains an operator. The program should add parentheses.
    let _ = Some(0).filter(|_| 1 == 0);
    //~^ some_filter

    let _ = Some(0).filter(|_| match 0 {
        //~^ some_filter
        0 => false,
        1 => true,
        _ => true,
    });

    // The argument to filter requires the value in the option. The program
    // can't figure out how to change it. It should leave it alone for now.
    let _ = Some(0).filter(|x| *x == 0);

    // The expression is a macro argument. The program should change the macro
    // argument. It should not expand the macro.
    let _ = unchanged!(Some(0).filter(|_| false));
    //~^ some_filter
    let _ = Some(0).filter(|_| vec![false].is_empty());
    //~^ some_filter

    // The condition is a macro that expands to an expression containing an
    // operator. The program should not add parentheses.
    let _ = Some(0).filter(|_| condition!(false));
    //~^ some_filter

    Some(String::from(
        //~^ some_filter
        "looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong",
    ))
    .filter(|_| 1 == 0);
    Some(5).filter(|_| {
        //~^ some_filter
        "looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong".is_empty()
    });
    Some(String::from(
        //~^ some_filter
        "looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong",
    ))
    .filter(|_| {
        "looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong".is_empty()
    });
}
