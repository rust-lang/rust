fn main() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: (), with a };
    //~^ ERROR expected one of `,` or `}`, found `a`
    //~| ERROR cannot find value `with` in this scope
    //~| ERROR struct `main::S` has no field named `with`
}
