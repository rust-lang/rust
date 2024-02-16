//@ run-pass
struct S {
    x: String
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
    let x = S { x: "Hello!".to_string() };
    x.foo();
}
