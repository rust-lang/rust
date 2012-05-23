// xfail-test

const foo: int = 4 >> 1;
enum bs { thing = foo }
fn main() { assert(thing as int == foo); }
