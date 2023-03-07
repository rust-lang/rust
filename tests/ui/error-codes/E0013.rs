static X: i32 = 42;
const Y: i32 = X; //~ ERROR constants cannot refer to statics [E0013]

fn main() {}
