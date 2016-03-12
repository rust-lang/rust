#![feature(plugin)]
#![plugin(clippy)]

#[deny(panic_params)]

fn missing() {
    if true {
        panic!("{}"); //~ERROR: You probably are missing some parameter
    } else {
        panic!("{:?}"); //~ERROR: You probably are missing some parameter
    }
}

fn ok_single() {
    panic!("foo bar");
}

fn ok_multiple() {
    panic!("{}", "This is {ok}");
}

fn ok_bracket() {
    // the match is just here because of #759, it serves no other purpose for the lint
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
}
