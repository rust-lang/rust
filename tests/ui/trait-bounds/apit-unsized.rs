fn foo(_: impl Iterator<Item = i32> + ?Sized) {} //~ ERROR [E0277]
fn bar(_: impl ?Sized) {} //~ ERROR [E0277]

fn main() {}
