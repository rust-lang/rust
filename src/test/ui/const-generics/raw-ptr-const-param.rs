// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Const<const P: *const u32>; //~ ERROR: using raw pointers as const generic parameters

fn main() {
    let _: Const<{ 15 as *const _ }> = Const::<{ 10 as *const _ }>;
    let _: Const<{ 10 as *const _ }> = Const::<{ 10 as *const _ }>;
}
