// compile-flags: --error-format pretty-json -Zunstable-options
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

    if (_b) { //~ ERROR
        println!("hello");
    }

    f();

}

fn f() -> bool {
    let c = false;

    if(c) { //~ ERROR
        println!("next");
    }

    if (c){ //~ ERROR
        println!("prev");
    }

    while (false && true){
        if (c) { //~ ERROR
            println!("norm");
        }

    }

    while(true && false) { //~ ERROR
        for _ in (0 .. 3){ //~ ERROR
            println!("e~")
        }
    }

    for _ in (0 .. 3) { //~ ERROR
        while (true && false) { //~ ERROR
            println!("e~")
        }
    }


    loop {
        if (break { return true }) {
        }
    }
    false
}
