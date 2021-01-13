#[cfg(FALSE)]
#[attr = multi::segment::path] //~ ERROR arbitrary expressions in key-value attributes are unstable
#[attr = macro_call!()] //~ ERROR arbitrary expressions in key-value attributes are unstable
#[attr = 1 + 2] //~ ERROR arbitrary expressions in key-value attributes are unstable
#[attr = what?] //~ ERROR arbitrary expressions in key-value attributes are unstable
struct S;

fn main() {}
