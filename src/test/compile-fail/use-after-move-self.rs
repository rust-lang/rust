struct S {
    x: ~int
}

pub impl S {
    fn foo(self) -> int {
        self.bar();
        return *self.x;  //~ ERROR use of moved value: `self`
    }

    fn bar(self) {}
}

fn main() {
    let x = S { x: ~1 };
    println(x.foo().to_str());
}
