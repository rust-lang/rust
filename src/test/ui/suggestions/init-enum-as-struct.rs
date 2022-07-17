pub type Foos = i32;

pub enum Foo {
    A,
    B,
    C,
}

mod module0 {
    pub struct Foo {
        a: i32,
    }
}
mod module1 {
    pub struct Foo {}
}
mod module2 {
    pub enum Foo {}
}

fn main() {
    let foo = Foo { b: 0 }; //~ expected struct, variant or union type, found enum `Foo`
}
