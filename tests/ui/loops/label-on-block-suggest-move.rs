// see https://github.com/rust-lang/rust/issues/138585
#![allow(break_with_label_and_loop)] // doesn't work locally

fn main() {
    loop 'a: {}
    //~^ ERROR: block label not supported here
    //~| HELP: if you meant to label the loop, move this label before the loop
    while false 'a: {}
    //~^ ERROR: block label not supported here
    //~| HELP: if you meant to label the loop, move this label before the loop
    for i in [0] 'a: {}
    //~^ ERROR: block label not supported here
    //~| HELP: if you meant to label the loop, move this label before the loop
    'a: loop {
        // first block is parsed as the break expr's value with or without parens
        while break 'a 'b: {} 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: if you meant to label the loop, move this label before the loop
        while break 'a ('b: {}) 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: if you meant to label the loop, move this label before the loop

        // without the parens, the first block is parsed as the while-loop's body
        // (see the 'no errors' section)
        // #[allow(break_with_label_and_loop)] (doesn't work locally)
        while (break 'a {}) 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: if you meant to label the loop, move this label before the loop
    }

    // do not suggest moving the label if there is already a label on the loop
    'a: loop 'b: {}
    //~^ ERROR: block label not supported here
    //~| HELP: remove this block label
    'a: while false 'b: {}
    //~^ ERROR: block label not supported here
    //~| HELP: remove this block label
    'a: for i in [0] 'b: {}
    //~^ ERROR: block label not supported here
    //~| HELP: remove this block label
    'a: loop {
        // first block is parsed as the break expr's value with or without parens
        'd: while break 'a 'b: {} 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: remove this block label
        'd: while break 'a ('b: {}) 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: remove this block label

        // without the parens, the first block is parsed as the while-loop's body
        // (see the 'no errors' section)
        // #[allow(break_with_label_and_loop)] (doesn't work locally)
        'd: while (break 'a {}) 'c: {}
        //~^ ERROR: block label not supported here
        //~| HELP: remove this block label
    }

    // no errors
    loop { 'a: {} }
    'a: loop { 'b: {} }
    while false { 'a: {} }
    'a: while false { 'b: {} }
    for i in [0] { 'a: {} }
    'a: for i in [0] { 'b: {} }
    'a: {}
    'a: { 'b: {} }
    'a: loop {
        // first block is parsed as the break expr's value if it is a labeled block
        while break 'a 'b: {} {}
        'd: while break 'a 'b: {} {}
        while break 'a ('b: {}) {}
        'd: while break 'a ('b: {}) {}
        // first block is parsed as the while-loop's body if it has no label
        // (the break expr is parsed as having no value),
        // so the second block is a normal stmt-block, and the label is allowed
        while break 'a {} 'c: {}
        while break 'a {} {}
        'd: while break 'a {} 'c: {}
        'd: while break 'a {} {}
    }

    // unrelated errors that should not be affected
    'a: 'b: {}
    //~^ ERROR: expected `while`, `for`, `loop` or `{` after a label
    //~| HELP: consider removing the label
    loop { while break 'b: {} {} }
    //~^ ERROR: parentheses are required around this expression to avoid confusion with a labeled break expression
    //~| HELP: wrap the expression in parentheses
    //~| ERROR: `break` or `continue` with no label in the condition of a `while` loop [E0590]
}
