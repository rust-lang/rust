type foo = {a: int, b: int};
type bar = {a: int, b: uint};

fn want_foo(f: foo) {}
fn have_bar(b: bar) {
    want_foo(b); //~ ERROR (in field `b`, int vs uint)
}

fn main() {}
