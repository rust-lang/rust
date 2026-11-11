fn main() {
    struct S {
        foo: (),
        bar: (),
    }

    let a = S { foo: (), bar: () };
    let b = S { foo: () with a, bar: () };
    //~^ ERROR expected one of `,`, `.`, `?`, `}`, or an operator, found `with`
}
