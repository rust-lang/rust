#![warn(clippy::for_loop_over_option, clippy::for_loop_over_result)]

/// Tests for_loop_over_result and for_loop_over_option

fn for_loop_over_option_and_result() {
    let option = Some(1);
    let result = option.ok_or("x not found");
    let v = vec![0, 1, 2];

    // check FOR_LOOP_OVER_OPTION lint
    for x in option {
        println!("{}", x);
    }

    // check FOR_LOOP_OVER_RESULT lint
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
