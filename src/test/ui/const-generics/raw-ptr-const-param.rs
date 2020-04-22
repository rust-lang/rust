#![feature(const_generics, const_compare_raw_pointers)]
//~^ WARN the feature `const_generics` is incomplete

struct Const<const P: *const u32>;

fn main() {
    let _: Const<{ 15 as *const _ }> = Const::<{ 10 as *const _ }>; //~ mismatched types
    let _: Const<{ 10 as *const _ }> = Const::<{ 10 as *const _ }>;
}
