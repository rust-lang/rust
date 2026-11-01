fn main() {
    h2(|_: (), _: (), _: (), x: &_| {});
    //~^ ERROR type mismatch in closure arguments
}

fn h2<F>(_: F)
where
    F: for<'t0> Fn(&(), Box<dyn Fn(&())>, &'t0 (), fn(&(), &())),
{
}
