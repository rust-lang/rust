#![warn(clippy::match_on_vec_items)]
#![allow(clippy::redundant_at_rest_pattern, clippy::useless_vec)]

fn match_with_wildcard() {
    let arr = vec![0, 1, 2, 3];
    let range = 1..3;
    let idx = 1;

    // Lint, may panic
    match arr[idx] {
        0 => println!("0"),
        1 => println!("1"),
        _ => {},
    }

    // Lint, may panic
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        _ => {},
    }
}

fn match_without_wildcard() {
    let arr = vec![0, 1, 2, 3];
    let range = 1..3;
    let idx = 2;

    // Lint, may panic
    match arr[idx] {
        0 => println!("0"),
        1 => println!("1"),
        num => {},
    }

    // Lint, may panic
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        [ref sub @ ..] => {},
    }
}

fn match_wildcard_and_action() {
    let arr = vec![0, 1, 2, 3];
    let range = 1..3;
    let idx = 3;

    // Lint, may panic
    match arr[idx] {
        0 => println!("0"),
        1 => println!("1"),
        _ => println!("Hello, World!"),
    }

    // Lint, may panic
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        _ => println!("Hello, World!"),
    }
}

fn match_vec_ref() {
    let arr = &vec![0, 1, 2, 3];
    let range = 1..3;
    let idx = 3;

    // Lint, may panic
    match arr[idx] {
        0 => println!("0"),
        1 => println!("1"),
        _ => {},
    }

    // Lint, may panic
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        _ => {},
    }
}

fn match_with_get() {
    let arr = vec![0, 1, 2, 3];
    let range = 1..3;
    let idx = 3;

    // Ok
    match arr.get(idx) {
        Some(0) => println!("0"),
        Some(1) => println!("1"),
        _ => {},
    }

    // Ok
    match arr.get(range) {
        Some(&[0, 1]) => println!("0 1"),
        Some(&[1, 2]) => println!("1 2"),
        _ => {},
    }
}

fn match_with_array() {
    let arr = [0, 1, 2, 3];
    let range = 1..3;
    let idx = 3;

    // Ok
    match arr[idx] {
        0 => println!("0"),
        1 => println!("1"),
        _ => {},
    }

    // Ok
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        _ => {},
    }
}

fn match_with_endless_range() {
    let arr = vec![0, 1, 2, 3];
    let range = ..;

    // Ok
    match arr[range] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        [0, 1, 2, 3] => println!("0, 1, 2, 3"),
        _ => {},
    }

    // Ok
    match arr[..] {
        [0, 1] => println!("0 1"),
        [1, 2] => println!("1 2"),
        [0, 1, 2, 3] => println!("0, 1, 2, 3"),
        _ => {},
    }
}

fn main() {
    match_with_wildcard();
    match_without_wildcard();
    match_wildcard_and_action();
    match_vec_ref();
    match_with_get();
    match_with_array();
    match_with_endless_range();
}
