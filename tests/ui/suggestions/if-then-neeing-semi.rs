// edition:2018

fn dummy() -> i32 {
    42
}

fn extra_semicolon() {
    let _ = if true {
        //~^ NOTE `if` and `else` have incompatible types
        dummy(); //~ NOTE expected because of this
        //~^ HELP consider removing this semicolon
    } else {
        dummy() //~ ERROR `if` and `else` have incompatible types
        //~^ NOTE expected `()`, found `i32`
    };
}

async fn async_dummy() {}

async fn async_dummy2() {}

async fn async_extra_semicolon_same() {
    let _ = if true {
        //~^ NOTE `if` and `else` have incompatible types
        async_dummy(); //~ NOTE expected because of this
        //~^ HELP consider removing this semicolon
    } else {
        async_dummy() //~ ERROR `if` and `else` have incompatible types
        //~^ NOTE expected `()`, found opaque type
        //~| NOTE expected unit type `()`
        //~| HELP consider `await`ing on the `Future`
    };
}

async fn async_extra_semicolon_different() {
    let _ = if true {
        //~^ NOTE `if` and `else` have incompatible types
        async_dummy(); //~ NOTE expected because of this
        //~^ HELP consider removing this semicolon
    } else {
        async_dummy2() //~ ERROR `if` and `else` have incompatible types
        //~^ NOTE expected `()`, found opaque type
        //~| NOTE expected unit type `()`
        //~| HELP consider `await`ing on the `Future`
    };
}

async fn async_different_futures() {
    let _ = if true {
        //~^ NOTE `if` and `else` have incompatible types
        async_dummy() //~ NOTE expected because of this
        //~| HELP consider `await`ing on both `Future`s
    } else {
        async_dummy2() //~ ERROR `if` and `else` have incompatible types
        //~^ NOTE expected opaque type, found a different opaque type
        //~| NOTE expected opaque type `impl Future<Output = ()>`
        //~| NOTE distinct uses of `impl Trait` result in different opaque types
    };
}

fn main() {}
