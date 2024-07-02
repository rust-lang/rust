#![warn(unused_labels)]

fn main() {
    'while_loop: while true { //~ WARN denote infinite loops with
        //~^ WARN unused label
        while_loop;
        //~^ ERROR cannot find value `while_loop`
    };
    'while_let: while let Some(_) = Some(()) {
        //~^ WARN unused label
        while_let;
        //~^ ERROR cannot find value `while_let`
    }
    'for_loop: for _ in 0..3 {
        //~^ WARN unused label
        for_loop;
        //~^ ERROR cannot find value `for_loop`
    };
    'LOOP: loop {
        //~^ WARN unused label
        LOOP;
        //~^ ERROR cannot find value `LOOP`
    };
}

fn foo() {
    'LOOP: loop {
        break LOOP;
        //~^ ERROR cannot find value `LOOP`
    };
    'while_loop: while true { //~ WARN denote infinite loops with
        break while_loop;
        //~^ ERROR cannot find value `while_loop`
    };
    'while_let: while let Some(_) = Some(()) {
        break while_let;
        //~^ ERROR cannot find value `while_let`
    }
    'for_loop: for _ in 0..3 {
        break for_loop;
        //~^ ERROR cannot find value `for_loop`
    };
}

fn bar() {
    let foo = ();
    'while_loop: while true { //~ WARN denote infinite loops with
        //~^ WARN unused label
        break foo;
        //~^ ERROR `break` with value from a `while` loop
    };
    'while_let: while let Some(_) = Some(()) {
        //~^ WARN unused label
        break foo;
        //~^ ERROR `break` with value from a `while` loop
    }
    'for_loop: for _ in 0..3 {
        //~^ WARN unused label
        break foo;
        //~^ ERROR `break` with value from a `for` loop
    };
}
