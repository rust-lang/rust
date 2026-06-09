//@ revisions: min adt_const_params full

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

struct Checked<const F: fn(usize) -> bool>;
//~^ ERROR: using function pointers as const generic parameters

fn not_one(val: usize) -> bool {
    val != 1
}
fn not_two(val: usize) -> bool {
    val != 2
}

fn generic_arg<T>(val: T) -> bool {
    true
}

fn generic<T>(val: usize) -> bool {
    val != 1
}

fn main() {
    let _: Option<Checked<not_one>> = None;
    let _: Checked<not_one> = Checked::<not_one>;
    let _: Checked<not_one> = Checked::<not_two>;

    let _ = Checked::<generic_arg>;
    let _ = Checked::<{ generic_arg::<usize> }>;
    let _ = Checked::<{ generic_arg::<u32> }>; //~ ERROR: mismatched types

    let _ = Checked::<generic>; //~ ERROR: type annotations needed
    let _ = Checked::<{ generic::<u16> }>;
    let _: Checked<{ generic::<u16> }> = Checked::<{ generic::<u16> }>;
    let _: Checked<{ generic::<u32> }> = Checked::<{ generic::<u16> }>;
}
