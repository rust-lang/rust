//@ compile-flags: -D while-true
//@ run-rustfix

fn main() {
    let mut i = 0;
    'a: while true { //~ ERROR denote infinite loops with `loop
        i += 1;
        if i == 5 { break 'a; }
    }
}
