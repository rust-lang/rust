mod foo {
    type T = ();
    struct S1(pub(in crate::foo) (), pub(T), pub(crate) (), pub(((), T)));
    struct S2(pub((foo)) ());
    //~^ ERROR expected one of `)` or `,`, found `(`
    //~| ERROR cannot find type `foo` in this scope
}

fn main() {}
