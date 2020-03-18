#![feature(const_generics)]
//~^ WARNING the feature `const_generics` is incomplete and may cause the compiler to crash

struct ConstFn<const F: fn()>;
//~^ ERROR using function pointers as const generic parameters is unstable

struct ConstPtr<const P: *const u32>;
//~^ ERROR using raw pointers as const generic parameters is unstable

fn main() {}
