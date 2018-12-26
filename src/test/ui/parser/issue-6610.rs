// compile-flags: -Z parse-only

trait Foo { fn a() } //~ ERROR expected `;` or `{`, found `}`

fn main() {}
