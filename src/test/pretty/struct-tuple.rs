// pp-exact
struct Foo;
struct Bar(int, int);

fn main() {
    struct Foo2;
    struct Bar2(int, int, int);
    let a = Bar(5, 5);
    let b = Foo;
}
