#![feature(const_generics, const_compare_raw_pointers)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Checked<const F: fn(usize) -> bool>;

fn not_one(val: usize) -> bool { val != 1 }
fn not_two(val: usize) -> bool { val != 2 }

fn generic_arg<T>(val: T) -> bool { true }

fn generic<T>(val: usize) -> bool { val != 1 }

fn main() {
    let _: Option<Checked<not_one>> = None;
    let _: Checked<not_one> = Checked::<not_one>;
    let _: Checked<not_one> = Checked::<not_two>; //~ mismatched types

    let _ = Checked::<generic_arg>;
    let _ = Checked::<{generic_arg::<usize>}>;
    let _ = Checked::<{generic_arg::<u32>}>;  //~ mismatched types

    let _ = Checked::<generic>; //~ type annotations needed
    let _ = Checked::<{generic::<u16>}>;
    let _: Checked<{generic::<u16>}> = Checked::<{generic::<u16>}>;
    let _: Checked<{generic::<u32>}> = Checked::<{generic::<u16>}>; //~ mismatched types
}
