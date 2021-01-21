#![warn(unused_labels)]

fn main() {
    'while_loop: while true { //~ WARN denote infinite loops with
        //~^ WARN unused label
        while_loop;
        //~^ ERROR cannot find value `while_loop` in this scope
    };
    'while_let: while let Some(_) = Some(()) {
        //~^ WARN unused label
        while_let;
        //~^ ERROR cannot find value `while_let` in this scope
    }
    'for_loop: for _ in 0..3 {
        //~^ WARN unused label
        for_loop;
        //~^ ERROR cannot find value `for_loop` in this scope
    };
    'LOOP: loop {
        //~^ WARN unused label
        LOOP;
        //~^ ERROR cannot find value `LOOP` in this scope
    };
}

fn foo() {
    'LOOP: loop {
        //~^ WARN unused label
        break LOOP;
        //~^ ERROR cannot find value `LOOP` in this scope
    };
    'while_loop: while true { //~ WARN denote infinite loops with
        //~^ WARN unused label
        break while_loop;
        //~^ ERROR cannot find value `while_loop` in this scope
    };
    'while_let: while let Some(_) = Some(()) {
        //~^ WARN unused label
        break while_let;
        //~^ ERROR cannot find value `while_let` in this scope
    }
    'for_loop: for _ in 0..3 {
        //~^ WARN unused label
        break for_loop;
        //~^ ERROR cannot find value `for_loop` in this scope
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
