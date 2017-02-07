#![feature(plugin)]
#![plugin(clippy)]

#![deny(panic_params)]

fn missing() {
    if true {
        panic!("{}"); //~ERROR: you probably are missing some parameter
    } else if false {
        panic!("{:?}"); //~ERROR: you probably are missing some parameter
    } else {
        assert!(true, "here be missing values: {}"); //~ERROR you probably are missing some parameter
    }
}

fn ok_single() {
    panic!("foo bar");
}

fn ok_inner() {
    // Test for #768
    assert!("foo bar".contains(&format!("foo {}", "bar")));
}

fn ok_multiple() {
    panic!("{}", "This is {ok}");
}

fn ok_bracket() {
    match 42 {
        1337 => panic!("{so is this"),
        666 => panic!("so is this}"),
        _ => panic!("}so is that{"),
    }
}

fn main() {
    missing();
    ok_single();
    ok_multiple();
    ok_bracket();
    ok_inner();
}
