// Here: foo is parameterized because it contains a method that
// refers to self.

iface foo {
    fn self_int() -> &self/int;

    fn any_int() -> &int;
}

type with_foo = {mut f: foo};

impl methods for with_foo {
    fn set_foo(f: foo) {
        self.f = f; //~ ERROR mismatched types: expected `foo/&self` but found `foo/&`
    }
}

// Bar is not region parameterized.

iface bar {
    fn any_int() -> &int;
}

type with_bar = {mut f: bar};

impl methods for with_bar {
    fn set_foo(f: bar) {
        self.f = f;
    }
}

fn main() {}