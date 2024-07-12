trait T { m!(); } //~ ERROR cannot find macro `m`

struct S;
impl S { m!(); } //~ ERROR cannot find macro `m`

fn main() {}
