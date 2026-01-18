struct A<T>(std::sync::OnceLock<Self>);
//~^ ERROR recursive type `A` has infinite size
//~| ERROR cycle detected when computing layout of `A`

static B: A<()> = todo!();
//~^ ERROR cycle occurred during layout computation

fn main() {}
