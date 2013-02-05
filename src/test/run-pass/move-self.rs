struct S {
    x: ~str
}

impl S {
    fn foo(self) {
        (move self).bar();
    }

    fn bar(self) {
        io::println(self.x);
    }
}

pub fn main() {
    let x = S { x: ~"Hello!" };
    x.foo();
}

