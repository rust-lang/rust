#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct Const<const P: *const u32>; //~ ERROR: using raw pointers as const generic parameters

fn main() {
    let _: Const<{ 15 as *const _ }> = Const::<{ 10 as *const _ }>;
    let _: Const<{ 10 as *const _ }> = Const::<{ 10 as *const _ }>;
}
