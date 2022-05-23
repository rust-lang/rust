#![feature(let_chains, let_else)]

fn main() {
    let opt = Some(1i32);

    let Some(n) = opt else {
        return;
    };
    let Some(n) = opt && n == 1 else {
    //~^ ERROR a `&&` expression cannot be directly assigned in `let...else`
    //~| ERROR mismatched types
    //~| ERROR mismatched types
        return;
    };
    let Some(n) = opt && let another = n else {
    //~^ ERROR a `&&` expression cannot be directly assigned in `let...else`
    //~| ERROR `let` expressions are not supported here
    //~| ERROR mismatched types
    //~| ERROR mismatched types
        return;
    };

    if let Some(n) = opt else {
    //~^ ERROR missing condition for `if` expression
        return;
    };
    if let Some(n) = opt && n == 1 else {
    //~^ ERROR missing condition for `if` expression
        return;
    };
    if let Some(n) = opt && let another = n else {
    //~^ ERROR missing condition for `if` expression
        return;
    };

    {
        while let Some(n) = opt else {
        //~^ ERROR expected `{`, found keyword `else`
            return;
        };
    }
    {
        while let Some(n) = opt && n == 1 else {
        //~^ ERROR expected `{`, found keyword `else`
            return;
        };
    }
    {
        while let Some(n) = opt && let another = n else {
        //~^ ERROR expected `{`, found keyword `else`
            return;
        };
    }
}
