struct ConstFn<const F: fn()>;
//~^ ERROR const generics are unstable
//~^^ ERROR use of function pointers as const generic arguments are unstable

struct ConstPtr<const P: *const u32>;
//~^ ERROR const generics are unstable
//~^^ ERROR use of raw pointers as const generic arguments are unstable

fn main() {}
