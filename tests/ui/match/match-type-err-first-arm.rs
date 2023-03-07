fn main() {
    let _ = test_func1(1);
    let _ = test_func2(1);
}

fn test_func1(n: i32) -> i32 { //~ NOTE expected `i32` because of return type
    match n {
        12 => 'b',
        //~^ ERROR mismatched types
        //~| NOTE expected `i32`, found `char`
        _ => 42,
    }
}

fn test_func2(n: i32) -> i32 {
    let x = match n { //~ NOTE `match` arms have incompatible types
        12 => 'b', //~ NOTE this is found to be of type `char`
        _ => 42,
        //~^ ERROR `match` arms have incompatible types
        //~| NOTE expected `char`, found integer
    };
    x
}

fn test_func3(n: i32) -> i32 {
    let x = match n { //~ NOTE `match` arms have incompatible types
        1 => 'b',
        2 => 'b',
        3 => 'b',
        4 => 'b',
        5 => 'b',
        6 => 'b',
        //~^ NOTE this and all prior arms are found to be of type `char`
        _ => 42,
        //~^ ERROR `match` arms have incompatible types
        //~| NOTE expected `char`, found integer
    };
    x
}

fn test_func4() {
    match Some(0u32) { //~ NOTE `match` arms have incompatible types
        Some(x) => {
            x //~ NOTE this is found to be of type `u32`
        },
        None => {}
        //~^ ERROR `match` arms have incompatible types
        //~| NOTE expected `u32`, found `()`
    };
}
