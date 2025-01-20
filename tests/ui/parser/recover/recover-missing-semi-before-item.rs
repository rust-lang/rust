//@ run-rustfix

#![allow(unused_variables, dead_code, unused_imports)]

fn for_struct() {
    let foo = 3 //~ ERROR expected `;`, found keyword `struct`
    struct Foo;
}

fn for_union() {
    let foo = 3 //~ ERROR expected `;`, found `union`
    union Foo {
        foo: usize,
    }
}

fn for_enum() {
    let foo = 3 //~ ERROR expected `;`, found keyword `enum`
    enum Foo {
        Bar,
    }
}

fn for_fn() {
    let foo = 3 //~ ERROR expected `;`, found keyword `fn`
    fn foo() {}
}

fn for_extern() {
    let foo = 3 //~ ERROR expected `;`, found keyword `extern`
    extern "C" fn foo() {}
}

fn for_impl() {
    struct Foo;
    let foo = 3 //~ ERROR expected `;`, found keyword `impl`
    impl Foo {}
}

fn for_use() {
    let foo = 3 //~ ERROR expected `;`, found keyword `pub`
    pub use bar::Bar;
}

fn for_mod() {
    let foo = 3 //~ ERROR expected `;`, found keyword `mod`
    mod foo {}
}

fn for_type() {
    let foo = 3 //~ ERROR expected `;`, found keyword `type`
    type Foo = usize;
}

mod bar {
    pub struct Bar;
}

const X: i32 = 123 //~ ERROR expected `;`, found keyword `fn`

fn main() {}
