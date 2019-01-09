trait T { m!(); } //~ ERROR cannot find macro `m!` in this scope

struct S;
impl S { m!(); } //~ ERROR cannot find macro `m!` in this scope

fn main() {}
