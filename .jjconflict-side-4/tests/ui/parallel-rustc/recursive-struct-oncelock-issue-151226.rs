// Test for #151226, Unable to verify registry association

struct A<T>(std::sync::OnceLock<Self>);
//~^ ERROR recursive type `A` has infinite size
static B: A<()> = todo!();
fn main() {}
