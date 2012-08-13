type foo = {a: int, b: int};
type bar = @foo;

fn want_foo(f: foo) {}
fn have_bar(b: bar) {
    want_foo(b); //~ ERROR (expected record but found @-ptr)
}

fn main() {}
