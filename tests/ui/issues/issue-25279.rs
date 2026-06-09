//@ run-pass
struct S<'a>(&'a ());

impl<'a> S<'a> {
    fn foo(self) -> &'a () {
        <Self>::bar(self)
    }

    fn bar(self) -> &'a () {
        self.0
    }
}

fn main() {
    S(&()).foo();
}
