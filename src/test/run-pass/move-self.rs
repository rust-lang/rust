struct S {
    x: ~str
}

impl S {
    pub fn foo(self) {
        self.bar();
    }

    pub fn bar(self) {
        println!("{}", self.x);
    }
}

pub fn main() {
    let x = S { x: ~"Hello!" };
    x.foo();
}
