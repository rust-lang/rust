#![warn(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![expect(clippy::branches_sharing_code, clippy::unnecessary_literal_unwrap)]

fn test_complex_conditions() {
    let x: Result<(), ()> = Ok(());
    let y: Result<(), ()> = Ok(());
    if x.is_ok() && y.is_err() {
        x.unwrap();
        //~^ unnecessary_unwrap

        x.unwrap_err();
        //~^ panicking_unwrap

        y.unwrap();
        //~^ panicking_unwrap

        y.unwrap_err();
        //~^ unnecessary_unwrap
    } else {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        x.unwrap_err();
        y.unwrap();
        y.unwrap_err();
    }

    if x.is_ok() || y.is_ok() {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
    } else {
        x.unwrap();
        //~^ panicking_unwrap

        x.unwrap_err();
        //~^ unnecessary_unwrap

        y.unwrap();
        //~^ panicking_unwrap

        y.unwrap_err();
        //~^ unnecessary_unwrap
    }
    let z: Result<(), ()> = Ok(());
    if x.is_ok() && !(y.is_ok() || z.is_err()) {
        x.unwrap();
        //~^ unnecessary_unwrap

        x.unwrap_err();
        //~^ panicking_unwrap

        y.unwrap();
        //~^ panicking_unwrap

        y.unwrap_err();
        //~^ unnecessary_unwrap

        z.unwrap();
        //~^ unnecessary_unwrap

        z.unwrap_err();
        //~^ panicking_unwrap
    }
    if x.is_ok() || !(y.is_ok() && z.is_err()) {
        // not statically determinable whether any of the following will always succeed or always fail:
        x.unwrap();
        y.unwrap();
        z.unwrap();
    } else {
        x.unwrap();
        //~^ panicking_unwrap

        x.unwrap_err();
        //~^ unnecessary_unwrap

        y.unwrap();
        //~^ unnecessary_unwrap

        y.unwrap_err();
        //~^ panicking_unwrap

        z.unwrap();
        //~^ panicking_unwrap

        z.unwrap_err();
        //~^ unnecessary_unwrap
    }
}

fn main() {}
