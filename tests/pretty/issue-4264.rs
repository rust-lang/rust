//@ pretty-compare-only
//@ pretty-mode:hir,typed
//@ pp-exact:issue-4264.pp

// #4264 fixed-length vector types

pub fn foo(_: [i32; 3]) {}

pub fn bar() {
    const FOO: usize = 5 - 4;
    let _: [(); FOO] = [()];

    let _ : [(); 1] = [()];

    let _ = &([1,2,3]) as *const _ as *const [i32; 3];

    format!("test");
}

pub type Foo = [i32; 3];

pub struct Bar {
    pub x: [i32; 3]
}

pub struct TupleBar([i32; 4]);

pub enum Baz {
    BazVariant([i32; 5])
}

pub fn id<T>(x: T) -> T { x }

pub fn use_id() {
    let _ = id::<[i32; 3]>([1,2,3]);
}


fn main() {}
