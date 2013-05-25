struct S {
    x: ~str
}

pub impl S {
    fn foo(self) {
        self.bar();
    }

    fn bar(self) {
        println(self.x);
    }
}

pub fn main() {
    let x = S { x: ~"Hello!" };
    x.foo();
}
