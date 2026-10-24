// Due to a compiler bug, if a return occurs outside of a function body
// (e.g. in an AnonConst body), the return value expression would not be
// type-checked, leading to an ICE. This test checks that the ICE no
// longer happens, and that an appropriate error message is issued that
// also explains why the return is considered "outside of a function body"
// if it seems to be inside one, as in the main function below.

const C: [(); 42] = {
    [(); return || {
        //~^ ERROR: return statement outside of function body [E0572]
        let tx;
    }]
};

struct S {}
trait Tr {
    fn foo();
    fn bar() {
        //~^ NOTE: ...not the enclosing function body
        [(); return];
        //~^ ERROR: return statement outside of function body [E0572]
        //~| NOTE: the return is part of this body...
    }
}
impl Tr for S {
    fn foo() {
        //~^ NOTE: ...not the enclosing function body
        [(); return];
        //~^ ERROR: return statement outside of function body [E0572]
        //~| NOTE: the return is part of this body...
    }
}

fn main() {
    //~^ NOTE: ...not the enclosing function body
    [(); return || {
        //~^ ERROR: return statement outside of function body [E0572]
        //~| NOTE: the return is part of this body...
        let tx;
    }];
}
