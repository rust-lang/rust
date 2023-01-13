trait Fun {
    type F<'a>: ?Sized;

    fn identity<'a>(t: &'a Self::F<'a>) -> &'a Self::F<'a> { t }
}

impl <T> Fun for T {
    type F<'a> = [u8];
}

fn bug<'a, T: ?Sized + Fun<F<'a> = [u8]>>(_ : Box<T>) -> &'static T::F<'a> {
    let a = [0; 1];
    let _x = T::identity(&a);
      //~^ ERROR: `a` does not live long enough
    todo!()
}


fn main() {
    let x = 10;

    bug(Box::new(x));
}
