// compile-flags: --error-format json
// run-rustfix

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

#![deny(unused_parens)]
#![allow(unreachable_code)]

fn main() {

    let _b = false;

    if (_b) { //~ ERROR unnecessary parentheses
        println!("hello");
    }

    f();

}

fn f() -> bool {
    let c = false;

    if(c) { //~ ERROR unnecessary parentheses
        println!("next");
    }

    if (c){ //~ ERROR unnecessary parentheses
        println!("prev");
    }

    while (false && true){
        if (c) { //~ ERROR unnecessary parentheses
            println!("norm");
        }

    }

    while(true && false) { //~ ERROR unnecessary parentheses
        for _ in (0 .. 3){ //~ ERROR unnecessary parentheses
            println!("e~")
        }
    }

    for _ in (0 .. 3) { //~ ERROR unnecessary parentheses
        while (true && false) { //~ ERROR unnecessary parentheses
            println!("e~")
        }
    }


    loop {
        if (break { return true }) {
        }
    }
    false
}
