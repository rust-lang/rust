//@ check-pass
//@ run-rustfix

fn main() {
    let mut _x: usize;
    _x = 1;
    println!("_x is {}", _x = 5);
    //~^ WARNING named argument `_x` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    println!("_x is {}", y = _x);
    //~^ WARNING named argument `y` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    println!("first positional arg {}, second positional arg {}, _x is {}", 1, 2, y = _x);
    //~^ WARNING named argument `y` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    let mut _x: usize;
    _x = 1;
    let _f = format!("_x is {}", _x = 5);
    //~^ WARNING named argument `_x` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    let _f = format!("_x is {}", y = _x);
    //~^ WARNING named argument `y` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity
    let _f = format!("first positional arg {}, second positional arg {}, _x is {}", 1, 2, y = _x);
    //~^ WARNING named argument `y` is not used by name [named_arguments_used_positionally]
    //~| HELP use the named argument by name to avoid ambiguity

    let s = "0.009";
    // Confirm that named arguments used in formatting are correctly considered.
    println!(".{:0<width$}", s, width = _x);

    let region = "abc";
    let width = 8;
    let ls = "abcde";
    let full = "abcde";
    // Confirm that named arguments used in formatting are correctly considered.
    println!(
        "| {r:rw$?} | {ui:4?} | {v}",
        r = region,
        rw = width,
        ui = ls,
        v = full,
    );

    // Confirm that named arguments used in formatting are correctly considered.
    println!("{:.a$}", "aaaaaaaaaaaaaaaaaa", a = 4);

    // Confirm that named arguments used in formatting are correctly considered.
    println!("{:._a$}", "aaaaaaaaaaaaaaaaaa", _a = 4);
}
