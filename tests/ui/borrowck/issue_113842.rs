mod some_library {
    pub fn foo(_: &mut [i32]) {}
    pub fn bar<'a>() -> &'a [i32] {
        &[]
    }
    pub fn bar_mut<'a>() -> &'a mut [i32] {
        &mut []
    }
}

struct Foo {
    pub x: i32,
}

fn foo() {
    let foo = Foo { x: 0 };
    let _y: &mut Foo = &mut &foo; //~ ERROR cannot borrow data in a `&` reference as mutable
}

fn main() {
    some_library::foo(&mut some_library::bar()); //~ ERROR cannot borrow data in a `&` reference as mutable
}
