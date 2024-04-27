#![allow(unused, dead_code)]

fn foo() -> u32 {
    return 'label: loop { break 'label 42; };
}

fn bar() -> u32 {
    loop { break 'label: loop { break 'label 42; }; }
    //~^ ERROR: parentheses are required around this expression to avoid confusion
    //~| HELP: wrap the expression in parentheses
}

fn baz() -> u32 {
    'label: loop {
        break 'label
        //~^ WARNING: this labeled break expression is easy to confuse with an unlabeled break
            loop { break 42; };
            //~^ HELP: wrap this expression in parentheses
    };

    'label2: loop {
        break 'label2 'inner: loop { break 42; };
        // no warnings or errors here
    }
}

pub fn main() {
    // Regression test for issue #86948, as resolved in #87026:
    let a = 'first_loop: loop {
        break 'first_loop 1;
    };
    let b = loop {
        break 'inner_loop: loop {
        //~^ ERROR: parentheses are required around this expression to avoid confusion
        //~| HELP: wrap the expression in parentheses
            break 'inner_loop 1;
        };
    };
}
