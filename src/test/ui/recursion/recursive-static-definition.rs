pub static FOO: u32 = FOO;
//~^ ERROR cycle detected when const-evaluating `FOO`

fn main() {}
