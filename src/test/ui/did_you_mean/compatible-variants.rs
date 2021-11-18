enum Hey<A, B> {
    A(A),
    B(B),
}

fn f() {}

fn a() -> Option<()> {
    while false {
        //~^ ERROR mismatched types
        f();
    }
    //~^ HELP try adding an expression
}

fn b() -> Result<(), ()> {
    f()
    //~^ ERROR mismatched types
    //~| HELP try adding an expression
}

fn main() {
    let _: Option<()> = while false {};
    //~^ ERROR mismatched types
    //~| HELP try wrapping
    let _: Option<()> = {
        while false {}
        //~^ ERROR mismatched types
        //~| HELP try adding an expression
    };
    let _: Result<i32, i32> = 1;
    //~^ ERROR mismatched types
    //~| HELP try wrapping
    let _: Option<i32> = 1;
    //~^ ERROR mismatched types
    //~| HELP try wrapping
    let _: Hey<i32, i32> = 1;
    //~^ ERROR mismatched types
    //~| HELP try wrapping
    let _: Hey<i32, bool> = false;
    //~^ ERROR mismatched types
    //~| HELP try wrapping
}
