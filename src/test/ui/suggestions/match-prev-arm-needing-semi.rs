// edition:2018

fn dummy() -> i32 { 42 }

fn extra_semicolon() {
    let _ = match true { //~ NOTE `match` arms have incompatible types
        true => {
            dummy(); //~ NOTE this is found to be
            //~^ HELP consider removing this semicolon
        }
        false => dummy(), //~ ERROR `match` arms have incompatible types
        //~^ NOTE expected `()`, found `i32`
    };
}

async fn async_dummy() {} //~ NOTE the `Output` of this `async fn`'s found opaque type

async fn async_extra_semicolon_same() {
    let _ = match true { //~ NOTE `match` arms have incompatible types
        true => {
            async_dummy(); //~ NOTE this is found to be
            //~^ HELP consider removing this semicolon
        }
        false => async_dummy(), //~ ERROR `match` arms have incompatible types
        //~^ NOTE expected `()`, found opaque type
        //~| NOTE expected type `()`
        //~| HELP consider `await`ing on the `Future`
    };
}

fn main() {}

