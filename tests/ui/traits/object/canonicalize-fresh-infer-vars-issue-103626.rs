trait TryFromBreak<R = <Self as Try>::Break> {
    fn from_break(residual: R) -> Self;
}

trait Try {
    type Break;
}

fn w<'a, T: 'a, F: Fn(&'a T)>() {
    let b: &dyn TryFromBreak = &();
    //~^ ERROR: the trait `TryFromBreak` is not dyn compatible
}

fn main() {}
