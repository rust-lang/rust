const fn and(a: bool, b: bool) -> bool { a && b }
//~^ ERROR `&&` is not allowed in a `const fn`
const fn or(a: bool, b: bool) -> bool { a || b }
//~^ ERROR `||` is not allowed in a `const fn`

fn main() {}
