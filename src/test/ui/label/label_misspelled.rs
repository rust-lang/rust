fn main() {
    'LOOP: loop {
        LOOP;
        //~^ ERROR cannot find value `LOOP` in this scope
    };
    'while_loop: while true { //~ WARN denote infinite loops with
        while_loop;
        //~^ ERROR cannot find value `while_loop` in this scope
    };
    'while_let: while let Some(_) = Some(()) {
        while_let;
        //~^ ERROR cannot find value `while_let` in this scope
    }
    'for_loop: for _ in 0..3 {
        for_loop;
        //~^ ERROR cannot find value `for_loop` in this scope
    };
}

fn foo() {
    'LOOP: loop {
        break LOOP;
        //~^ ERROR cannot find value `LOOP` in this scope
    };
    'while_loop: while true { //~ WARN denote infinite loops with
        break while_loop;
        //~^ ERROR cannot find value `while_loop` in this scope
        //~| ERROR `break` with value from a `while` loop
    };
    'while_let: while let Some(_) = Some(()) {
        break while_let;
        //~^ ERROR cannot find value `while_let` in this scope
        //~| ERROR `break` with value from a `while` loop
    }
    'for_loop: for _ in 0..3 {
        break for_loop;
        //~^ ERROR cannot find value `for_loop` in this scope
        //~| ERROR `break` with value from a `for` loop
    };
}
