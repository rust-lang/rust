struct S;

trait D {
    type P<T: Copy>;
    //~^ NOTE required by this bound in `D::P`
    //~| NOTE required by a bound in `D::P`
}

impl D for S {
    type P<T: Copy> = ();
}

fn main() {
    let _: <S as D>::P<String>;
    //~^ ERROR the trait bound `String: Copy` is not satisfied
    //~| NOTE the trait `Copy` is not implemented for `String`
}
