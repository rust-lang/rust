// error-pattern:bad input

fn main() {
    Some("foo").unwrap_or(panic!("bad input")).to_string();
}
