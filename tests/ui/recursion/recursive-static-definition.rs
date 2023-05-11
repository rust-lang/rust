pub static FOO: u32 = FOO;
//~^ ERROR cycle detected when const-evaluating + checking `FOO`

fn main() {}
