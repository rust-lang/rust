fn foo(_: &'static [&'static str]) {}
fn bar(_: &'static [&'static str; 3]) {}

fn main() {
    foo(&["a", "b", "c"]);
    bar(&["d", "e", "f"]);
}
