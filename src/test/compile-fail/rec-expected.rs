type foo = {a: int};
type bar = {b: int};

fn want_foo(f: foo) {}
fn have_bar(b: bar) {
    want_foo(b); //! ERROR expected a record with field `a`
}

fn main() {}