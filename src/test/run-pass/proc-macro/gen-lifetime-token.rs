// aux-build:gen-lifetime-token.rs

extern crate gen_lifetime_token as bar;

bar::bar!();

fn main() {
    let x: &'static i32 = FOO;
    assert_eq!(*x, 1);
}
