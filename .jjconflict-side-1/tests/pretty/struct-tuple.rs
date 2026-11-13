//@ pp-exact
struct Foo;
struct Bar(isize, isize);

fn main() {
    struct Foo2;
    struct Bar2(isize, isize, isize);
    let _a = Bar(5, 5);
    let _b = Foo;
}
