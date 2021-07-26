// issue-49296: Unsafe shenigans in constants can result in missing errors

#![feature(const_fn_trait_bound)]
#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

const unsafe fn transmute<T: ?const Copy, U: ?const Copy>(t: T) -> U {
    #[repr(C)]
    union Transmute<T: Copy, U: Copy> {
        from: T,
        to: U,
    }

    Transmute { from: t }.to
}

const fn wat(x: u64) -> &'static u64 {
    unsafe { transmute(&x) }
}
const X: u64 = *wat(42);
//~^ ERROR evaluation of constant value failed

fn main() {
    println!("{}", X);
}
