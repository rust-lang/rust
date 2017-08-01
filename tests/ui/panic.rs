#![feature(plugin)]
#![plugin(clippy)]

#![warn(panic_params)]

fn missing() {
    if true {
        panic!("{}");
    } else if false {
        panic!("{:?}");
    } else {
        assert!(true, "here be missing values: {}");
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
