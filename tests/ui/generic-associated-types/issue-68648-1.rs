//@ check-pass

trait Fun {
    type F<'a>;

    fn identity<'a>(t: Self::F<'a>) -> Self::F<'a> { t }
}

impl <T> Fun for T {
    type F<'a> = Self;
}

fn bug<'a, T: for<'b> Fun<F<'b> = T>>(t: T) -> T::F<'a> {
    T::identity(t)
}


fn main() {
    let x = 10;

    bug(x);
}
