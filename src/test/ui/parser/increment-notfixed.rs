struct Foo {
    bar: Bar,
}

struct Bar {
    qux: i32,
}

fn post_field() {
    let foo = Foo { bar: Bar { qux: 0 } };
    foo.bar.qux++;
    //~^ ERROR Rust has no postfix increment operator
    println!("{}", foo.bar.qux);
}

fn pre_field() {
    let foo = Foo { bar: Bar { qux: 0 } };
    ++foo.bar.qux;
    //~^ ERROR Rust has no prefix increment operator
    println!("{}", foo.bar.qux);
}

fn main() {
    post_field();
    pre_field();
}
