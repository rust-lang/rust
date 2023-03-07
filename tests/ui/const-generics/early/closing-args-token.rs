struct S<const X: u32>;
struct T<const X: bool>;

fn bad_args_1() {
    S::<5 + 2 >> 7>;
    //~^ ERROR expressions must be enclosed in braces to be used as const generic arguments
    //~| ERROR comparison operators cannot be chained
}

fn bad_args_2() {
    S::<{ 5 + 2 } >> 7>;
    //~^ ERROR comparison operators cannot be chained
}

fn bad_args_3() {
    T::<0 >= 3>;
    //~^ ERROR expected expression, found `;`
}

fn bad_args_4() {
    let mut x = 0;
    T::<x >>= 2 > 0>;
    //~^ ERROR comparison operators cannot be chained
}

fn main() {}
