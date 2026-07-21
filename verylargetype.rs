#![recursion_limit = "10"]
struct B<T: 'static>(&'static B<(T, T)>, T);

static BOO: B<u32> = todo!();
