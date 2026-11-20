trait A<C = <Self as D>::E> {}

trait D {
    type E;
}

impl A<()> for () {}
impl D for () {
    type E = ();
}

fn f() {
    let B: &dyn A = &();
    //~^ ERROR the type parameter `C` must be explicitly specified
}

fn main() {}
