//@ compile-flags: -Zverbose-internals

mod mod1 {
    pub fn f<T: std::fmt::Display>(t: T)
    {
        let x = 20;

        let c = || println!("{} {}", t, x);
        let c1 : () = c;
        //~^ ERROR mismatched types
    }
}

fn main() {
    mod1::f(5i32);
}
