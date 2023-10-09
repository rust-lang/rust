pub static FOO: u32 = FOO;
//~^ ERROR cycle detected when evaluating initializer of static `FOO`

fn main() {}
