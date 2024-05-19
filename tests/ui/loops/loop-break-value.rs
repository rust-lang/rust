#![feature(never_type)]

fn main() {
    let val: ! = loop { break break; };
    //~^ ERROR mismatched types

    loop {
        if true {
            break "asdf";
        } else {
            break 123; //~ ERROR mismatched types
        }
    };

    let _: i32 = loop {
        break "asdf"; //~ ERROR mismatched types
    };

    let _: i32 = 'outer_loop: loop {
        loop {
            break 'outer_loop "nope"; //~ ERROR mismatched types
            break "ok";
        };
    };

    let _: Option<String> = loop {
        break; //~ ERROR mismatched types
    };

    'while_loop: while true { //~ WARN denote infinite loops with
        break;
        break (); //~ ERROR `break` with value from a `while` loop
        loop {
            break 'while_loop 123;
            //~^ ERROR `break` with value from a `while` loop
            break 456;
            break 789;
        };
    }

    while let Some(_) = Some(()) {
        if break () { //~ ERROR `break` with value from a `while` loop
        }
    }

    while let Some(_) = Some(()) {
        break None;
        //~^ ERROR `break` with value from a `while` loop
    }

    'while_let_loop: while let Some(_) = Some(()) {
        loop {
            break 'while_let_loop "nope";
            //~^ ERROR `break` with value from a `while` loop
            break 33;
        };
    }

    for _ in &[1,2,3] {
        break (); //~ ERROR `break` with value from a `for` loop
        break [()];
        //~^ ERROR `break` with value from a `for` loop
    }

    'for_loop: for _ in &[1,2,3] {
        loop {
            break Some(3);
            break 'for_loop Some(17);
            //~^ ERROR `break` with value from a `for` loop
        };
    }

    let _: i32 = 'a: loop {
        let _: () = 'b: loop {
            break ('c: loop {
                break;
                break 'c 123; //~ ERROR mismatched types
            });
            break 'a 123;
        };
    };

    loop {
        break (break, break); //~ ERROR mismatched types
    };

    loop {
        break;
        break 2; //~ ERROR mismatched types
    };

    loop {
        break 2;
        break; //~ ERROR mismatched types
        break 4;
    };

    'LOOP: for _ in 0 .. 9 {
        break LOOP;
        //~^ ERROR cannot find value `LOOP` in this scope
    }

    let _ = 'a: loop {
        loop {
            break; // This doesn't affect the expected break type of the 'a loop
            loop {
                loop {
                    break 'a 1;
                }
            }
        }
        break; //~ ERROR mismatched types
    };

    let _ = 'a: loop {
        loop {
            break; // This doesn't affect the expected break type of the 'a loop
            loop {
                loop {
                    break 'a 1;
                }
            }
        }
        break 'a; //~ ERROR mismatched types
    };

    loop {
        break;
        let _ = loop {
            break 2;
            loop {
                break;
            }
        };
        break 2; //~ ERROR mismatched types
    }

    'a: loop {
        break;
        let _ = 'a: loop {
            //~^ WARNING label name `'a` shadows a label name that is already in scope
            break 2;
            loop {
                break 'a; //~ ERROR mismatched types
            }
        };
        break 2; //~ ERROR mismatched types
    }

    'a: loop {
        break;
        let _ = 'a: loop {
            //~^ WARNING label name `'a` shadows a label name that is already in scope
            break 'a 2;
            loop {
                break 'a; //~ ERROR mismatched types
            }
        };
        break 2; //~ ERROR mismatched types
    };

    loop { // point at the return type
        break 2; //~ ERROR mismatched types
    }
}
