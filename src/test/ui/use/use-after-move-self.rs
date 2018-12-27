#![feature(box_syntax)]

struct S {
    x: Box<isize>,
}

impl S {
    pub fn foo(self) -> isize {
        self.bar();
        return *self.x;  //~ ERROR use of moved value: `self`
    }

    pub fn bar(self) {}
}

fn main() {
    let x = S { x: box 1 };
    println!("{}", x.foo());
}
