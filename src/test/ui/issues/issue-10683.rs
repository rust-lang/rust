// run-pass
// pretty-expanded FIXME #23616

static NAME: &'static str = "hello world";

fn main() {
    match &*NAME.to_ascii_lowercase() {
        "foo" => {}
        _ => {}
    }
}
