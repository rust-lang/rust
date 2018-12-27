// issue-49296: Unsafe shenigans in constants can result in missing errors

#![feature(const_fn)]
#![feature(const_fn_union)]

const unsafe fn transmute<T: Copy, U: Copy>(t: T) -> U {
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
//~^ ERROR any use of this value will cause an error

fn main() {
    println!("{}", X);
}
