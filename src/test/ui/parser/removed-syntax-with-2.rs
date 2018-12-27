// compile-flags: -Z parse-only

fn removed_with() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: (), with a };
    //~^ ERROR expected one of `,` or `}`, found `a`
}
