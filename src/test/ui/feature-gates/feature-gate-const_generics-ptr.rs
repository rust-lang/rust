struct ConstFn<const F: fn()>;
//~^ ERROR const generics are unstable
//~^^ ERROR using function pointers as const generic parameters is forbidden

struct ConstPtr<const P: *const u32>;
//~^ ERROR const generics are unstable
//~^^ ERROR using raw pointers as const generic parameters is forbidden

fn main() {}
