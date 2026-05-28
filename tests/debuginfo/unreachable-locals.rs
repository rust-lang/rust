//@ compile-flags:-g
//@ disable-gdb-pretty-printers

#![allow(unused_variables)]

// No need to actually run the debugger, just make sure that the compiler can
// handle locals in unreachable code.

fn after_return() {
    return;
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in &[111i32] {}
    let test = if some_predicate() { 1 } else { 2 };
    while some_predicate() {
        let abc = !some_predicate();
    }
    loop {
        let abc = !some_predicate();
        break;
    }
    // nested block
    {
        let abc = !some_predicate();

        {
            let def = !some_predicate();
        }
    }
}

fn after_panic() {
    panic!();
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in &[111i32] {}
    let test = if some_predicate() { 1 } else { 2 };
    while some_predicate() {
        let abc = !some_predicate();
    }
    loop {
        let abc = !some_predicate();
        break;
    }
    // nested block
    {
        let abc = !some_predicate();

        {
            let def = !some_predicate();
        }
    }
}

fn after_diverging_function() {
    diverge();
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in &[111i32] {}
    let test = if some_predicate() { 1 } else { 2 };
    while some_predicate() {
        let abc = !some_predicate();
    }
    loop {
        let abc = !some_predicate();
        break;
    }
    // nested block
    {
        let abc = !some_predicate();

        {
            let def = !some_predicate();
        }
    }
}

fn after_break() {
    loop {
        break;
        let x = "0";
        let (ref y,z) = (1i32, 2u32);
        match (20i32, 'c') {
            (a, ref b) => {}
        }
        for a in &[111i32] {}
        let test = if some_predicate() { 1 } else { 2 };
        while some_predicate() {
            let abc = !some_predicate();
        }
        loop {
            let abc = !some_predicate();
            break;
        }
        // nested block
        {
            let abc = !some_predicate();

            {
                let def = !some_predicate();
            }
        }
    }
}

fn after_continue() {
    for _ in 0..10i32 {
        continue;
        let x = "0";
        let (ref y,z) = (1i32, 2u32);
        match (20i32, 'c') {
            (a, ref b) => {}
        }
        for a in &[111i32] {}
        let test = if some_predicate() { 1 } else { 2 };
        while some_predicate() {
            let abc = !some_predicate();
        }
        loop {
            let abc = !some_predicate();
            break;
        }
        // nested block
        {
            let abc = !some_predicate();

            {
                let def = !some_predicate();
            }
        }
    }
}

fn main() {
    after_return();
    after_panic();
    after_diverging_function();
    after_break();
    after_continue();
}

fn diverge() -> ! {
    panic!();
}

fn some_predicate() -> bool { true || false }
