struct S<A, B>(Option<(A, B)>);

impl<A, B> S<A, B> {
    fn infer(&self, a: A, b: B) {}
    //~^ NOTE associated function defined here
    //~| NOTE
    //~| NOTE
}

fn main() {
    let s = S(None);
    s.infer(0i32);
    //~^ ERROR this method takes 2 arguments but 1 argument was supplied
    //~| NOTE an argument is missing
    //~| HELP provide the argument
    let t: S<u32, _> = s;
    //~^ ERROR mismatched types
    //~| NOTE expected `S<u32, _>`, found `S<i32, _>`
    //~| NOTE expected due to this
    //~| NOTE expected struct `S<u32, _>`
}
