struct S {
    x: int,
    drop {}
}

impl S {
    fn foo(self) -> int {
        self.bar();
        return self.x;  //~ ERROR use of moved variable
    }

    fn bar(self) {}
}

fn main() {
    let x = S { x: 1 };
    io::println(x.foo().to_str());
}

