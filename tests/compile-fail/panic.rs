#![feature(plugin)]
#![plugin(clippy)]

#[deny(panic_params)]

fn missing() {
    panic!("{}"); //~ERROR: You probably are missing some parameter
}

fn ok_sigle() {
    panic!("foo bar");
}

fn ok_multiple() {
    panic!("{}", "This is {ok}");
}

fn main() {
    missing();
    ok_sigle();
    ok_multiple();
}
