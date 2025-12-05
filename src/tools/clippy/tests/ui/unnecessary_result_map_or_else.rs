#![warn(clippy::unnecessary_result_map_or_else)]
#![allow(clippy::unnecessary_literal_unwrap, clippy::let_and_return, clippy::let_unit_value)]

fn main() {
    let x: Result<(), ()> = Ok(());
    x.map_or_else(|err| err, |n| n);
    //~^ unnecessary_result_map_or_else

    // Type ascribtion.
    let x: Result<(), ()> = Ok(());
    x.map_or_else(|err: ()| err, |n: ()| n);
    //~^ unnecessary_result_map_or_else

    // Auto-deref.
    let y = String::new();
    let x: Result<&String, &String> = Ok(&y);
    let y: &str = x.map_or_else(|err| err, |n| n);
    //~^ unnecessary_result_map_or_else

    // Temporary variable.
    let x: Result<(), ()> = Ok(());
    x.map_or_else(
        //~^ unnecessary_result_map_or_else
        |err| err,
        |n| {
            let tmp = n;
            let tmp2 = tmp;
            tmp2
        },
    );

    // Should not warn.
    let x: Result<usize, usize> = Ok(0);
    x.map_or_else(|err| err, |n| n + 1);

    // Should not warn.
    let y = ();
    let x: Result<(), ()> = Ok(());
    x.map_or_else(|err| err, |_| y);

    // Should not warn.
    let y = ();
    let x: Result<(), ()> = Ok(());
    x.map_or_else(
        |err| err,
        |_| {
            let tmp = y;
            tmp
        },
    );

    // Should not warn.
    let x: Result<usize, usize> = Ok(1);
    x.map_or_else(
        |err| err,
        |n| {
            let tmp = n + 1;
            tmp
        },
    );

    // Should not warn.
    let y = 0;
    let x: Result<usize, usize> = Ok(1);
    x.map_or_else(
        |err| err,
        |n| {
            let tmp = n;
            y
        },
    );
}
