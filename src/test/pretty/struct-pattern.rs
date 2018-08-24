// pp-exact
// pretty-compare-only
// Testing that shorthand struct patterns are preserved

fn main() { let Foo { a, ref b, mut c, x: y, z: z } = foo; }
