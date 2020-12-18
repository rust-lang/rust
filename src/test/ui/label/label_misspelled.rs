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
