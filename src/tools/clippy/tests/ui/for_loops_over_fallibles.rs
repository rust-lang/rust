#![warn(clippy::for_loops_over_fallibles)]

fn for_loops_over_fallibles() {
    let option = Some(1);
    let result = option.ok_or("x not found");
    let v = vec![0, 1, 2];

    // check over an `Option`
    for x in option {
        println!("{}", x);
    }

    // check over a `Result`
    for x in result {
        println!("{}", x);
    }

    for x in option.ok_or("x not found") {
        println!("{}", x);
    }

    // make sure LOOP_OVER_NEXT lint takes clippy::precedence when next() is the last call
    // in the chain
    for x in v.iter().next() {
        println!("{}", x);
    }

    // make sure we lint when next() is not the last call in the chain
    for x in v.iter().next().and(Some(0)) {
        println!("{}", x);
    }

    for x in v.iter().next().ok_or("x not found") {
        println!("{}", x);
    }

    // check for false positives

    // for loop false positive
    for x in v {
        println!("{}", x);
    }

    // while let false positive for Option
    while let Some(x) = option {
        println!("{}", x);
        break;
    }

    // while let false positive for Result
    while let Ok(x) = result {
        println!("{}", x);
        break;
    }
}

fn main() {}
