//@ edition: 2024

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
    //~| ERROR mismatched types
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
        return;
    };

    if let Some(n) = opt else {
    //~^ ERROR this `if` expression is missing a block after the condition
        return;
    };
    if let Some(n) = opt && n == 1 else {
    //~^ ERROR this `if` expression is missing a block after the condition
        return;
    };
    if let Some(n) = opt && let another = n else {
    //~^ ERROR this `if` expression is missing a block after the condition
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
