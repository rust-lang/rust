#![feature(never_type)]

enum Helper<T, U> {
    T(T, [!; 0]),
    #[allow(dead_code)]
    U(U),
}

fn transmute<T, U>(t: T) -> U {
    let Helper::U(u) = Helper::T(t, []);
    //~^ ERROR refutable pattern in local binding: `T(_, _)` not covered
    u
    //~^ WARN use of possibly uninitialized variable: `u`
    //~| WARN this error has been downgraded to a warning for backwards compatibility
    //~| WARN this represents potential undefined behavior in your code and this warning will
}

fn main() {
    println!("{:?}", transmute::<&str, (*const u8, u64)>("type safety"));
}
