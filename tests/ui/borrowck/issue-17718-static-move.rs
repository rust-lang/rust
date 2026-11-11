struct Foo;
const INIT: Foo = Foo;
static FOO: Foo = INIT;

fn main() {
    let _a = FOO; //~ ERROR: cannot move out of static item
}
