#![feature(generic_associated_types)]

trait Fun {
    type F<'a>;
    
    fn identity<'a>(t: Self::F<'a>) -> Self::F<'a> { t }
}

impl <T> Fun for T {
    type F<'a> = Self;
}

fn bug<'a, T: for<'b> Fun<F<'b> = ()>>(t: T) -> T::F<'a> {
    T::identity(())
    //~^ ERROR: type mismatch resolving `for<'b> <{integer} as Fun>::F<'b> == ()`
}


fn main() {
    let x = 10;
    
    bug(x);
}
