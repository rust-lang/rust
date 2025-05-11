struct S<A, B>(Option<(A, B)>);

impl<A, B> S<A, B> {
    fn infer(&self, a: A, b: B) {}
    //~^ NOTE method defined here
}

fn main() {
    let s = S(None);
    s.infer(0i32);
    //~^ ERROR this method takes 2 arguments but 1 argument was supplied
    //~| NOTE this argument has type `i32`...
    //~| NOTE ... which causes `s` to have type `S<i32, _>`
    //~| NOTE argument #2 is missing
    //~| HELP provide the argument
    //~| HELP change the type of the numeric literal from `i32` to `u32`
    let t: S<u32, _> = s;
    //~^ ERROR mismatched types
    //~| NOTE expected `S<u32, _>`, found `S<i32, _>`
    //~| NOTE expected due to this
    //~| NOTE expected struct `S<u32, _>`
}
