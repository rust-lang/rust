//@ run-rustfix

struct Foo {
    bar: Bar,
}

struct Bar {
    qux: i32,
}

pub fn post_regular() {
    let mut i = 0;
    i++; //~ ERROR Rust has no postfix increment operator
    println!("{}", i);
}

pub fn post_while() {
    let mut i = 0;
    while i++ < 5 {
        //~^ ERROR Rust has no postfix increment operator
        println!("{}", i);
    }
}

pub fn post_regular_tmp() {
    let mut tmp = 0;
    tmp++; //~ ERROR Rust has no postfix increment operator
    println!("{}", tmp);
}

pub fn post_while_tmp() {
    let mut tmp = 0;
    while tmp++ < 5 {
        //~^ ERROR Rust has no postfix increment operator
        println!("{}", tmp);
    }
}

pub fn post_field() {
    let mut foo = Foo { bar: Bar { qux: 0 } };
    foo.bar.qux++;
    //~^ ERROR Rust has no postfix increment operator
    println!("{}", foo.bar.qux);
}

pub fn post_field_tmp() {
    struct S {
        tmp: i32
    }
    let mut s = S { tmp: 0 };
    s.tmp++;
    //~^ ERROR Rust has no postfix increment operator
    println!("{}", s.tmp);
}

pub fn pre_field() {
    let mut foo = Foo { bar: Bar { qux: 0 } };
    ++foo.bar.qux;
    //~^ ERROR Rust has no prefix increment operator
    println!("{}", foo.bar.qux);
}

fn main() {}
