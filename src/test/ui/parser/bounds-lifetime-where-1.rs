// compile-flags: -Z parse-only

type A where 'a; //~ ERROR expected `:`, found `;`

fn main() {}
