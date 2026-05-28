#![feature(never_type)]

fn main() {
    // The `if false` expressions are simply to
    // make sure we don't avoid checking everything
    // simply because a few expressions are unreachable.

    if false {
        let _: ! = { //~ ERROR mismatched types
            'a: while break 'a {};
        };
    }

    if false {
        let _: ! = {
            while false { //~ ERROR mismatched types
                break
            }
        };
    }

    if false {
        let _: ! = {
            while false { //~ ERROR mismatched types
                return
            }
        };
    }
}
