fn main() {
    // The `if false` expressions are simply to
    // make sure we don't avoid checking everything
    // simply because a few expressions are unreachable.

    if false {
        let _: ! = {
            'a: while break 'a {}
            //~^ ERROR mismatched types
        };
    }

    if false {
        let _: ! = {
            while false {
                //~^ ERROR mismatched types
                break;
            }
        };
    }

    if false {
        let _: ! = {
            while false {
                //~^ ERROR mismatched types
                return;
            }
        };
    }
}
