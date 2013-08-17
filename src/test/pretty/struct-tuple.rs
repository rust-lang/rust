// pp-exact
struct Foo;
struct Bar(int, int);

fn main() {
    struct Foo2;
    struct Bar2(int, int, int);
    let _a = Bar(5, 5);
    let _b = Foo;
}
