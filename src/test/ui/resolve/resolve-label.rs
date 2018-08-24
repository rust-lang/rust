fn f() {
    'l: loop {
        fn g() {
            loop {
                break 'l; //~ ERROR use of undeclared label
            }
        }
    }

    loop { 'w: while break 'w { } }
}

fn main() {}
