// Regression test for #122561

fn for_infinite() -> bool {
    for i in 0.. {
    //~^ ERROR mismatched types
        return false;
    }
}

fn for_finite() -> String {
    for i in 0..5 {
    //~^ ERROR mismatched types
        return String::from("test");
    }
}

fn for_zero_times() -> bool {
    for i in 0..0 {
    //~^ ERROR mismatched types
        return true;
    }
}

fn for_never_type() -> ! {
    for i in 0..5 {
    //~^ ERROR mismatched types
    }
}

// Entire function on a single line.
// Tests that we format the suggestion
// correctly in this case
fn for_single_line() -> bool { for i in 0.. { return false; } }
//~^ ERROR mismatched types

// Loop in an anon const in function args
// Tests that we:
// a. deal properly with this complex case
// b. format the suggestion correctly so
//    that it's readable
fn for_in_arg(a: &[(); for x in 0..2 {}]) -> bool {
    //~^ ERROR mismatched types
    true
}

fn while_inifinite() -> bool {
    while true {
    //~^ ERROR mismatched types
    //~| WARN denote infinite loops with `loop { ... }` [while_true]
        return true;
    }
}

fn while_finite() -> bool {
    let mut i = 0;
    while i < 3 {
    //~^ ERROR mismatched types
        i += 1;
        return true;
    }
}

fn while_zero_times() -> bool {
    while false {
    //~^ ERROR mismatched types
        return true;
    }
}

fn while_never_type() -> ! {
    while true {
    //~^ ERROR mismatched types
    //~| WARN denote infinite loops with `loop { ... }` [while_true]
    }
}

// No type mismatch error in this case
fn loop_() -> bool {
    loop {
        return true;
    }
}

const C: i32 = {
    for i in 0.. {
    //~^ ERROR mismatched types
    }
};

fn main() {
    let _ = [10; {
        for i in 0..5 {
        //~^ ERROR mismatched types
        }
    }];

    let _ = [10; {
        while false {
        //~^ ERROR mismatched types
        }
    }];


    let _ = |a: &[(); for x in 0..2 {}]| {};
    //~^ ERROR mismatched types
}
