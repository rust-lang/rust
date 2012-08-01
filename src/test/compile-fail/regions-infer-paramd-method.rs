// Here: foo is parameterized because it contains a method that
// refers to self.

trait foo {
    fn self_int() -> &self/int;

    fn any_int() -> &int;
}

type with_foo = {mut f: foo};

trait set_foo_foo {
    fn set_foo(f: foo);
}

impl methods of set_foo_foo for with_foo {
    fn set_foo(f: foo) {
        self.f = f; //~ ERROR mismatched types: expected `foo/&self` but found `foo/&`
    }
}

// Bar is not region parameterized.

trait bar {
    fn any_int() -> &int;
}

type with_bar = {mut f: bar};

trait set_foo_bar {
    fn set_foo(f: bar);
}

impl methods of set_foo_bar for with_bar {
    fn set_foo(f: bar) {
        self.f = f;
    }
}

fn main() {}
