//@ needs-rustc-debug-assertions

struct NonGeneric {}

#[derive(Default)]
//~^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument was supplied
//~^^ ERROR struct takes 0 lifetime arguments but 1 lifetime argument was supplied
//~^^^ ERROR struct takes 0 generic arguments but 1 generic argument was supplied
//~^^^^ ERROR struct takes 0 generic arguments but 1 generic argument was supplied
struct NonGeneric<'a, const N: usize> {}
//~^ ERROR lifetime parameter `'a` is never used
//~^^ ERROR the name `NonGeneric` is defined multiple times

pub fn main() {}
