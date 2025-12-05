fn main() {
    //~^ NOTE: not the enclosing function body
    //~| NOTE: not the enclosing function body
    //~| NOTE: not the enclosing function body
    //~| NOTE: not the enclosing function body
    |_: [_; return || {}]| {};
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...

    [(); return || {}];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...

    [(); return |ice| {}];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...

    [(); return while let Some(n) = Some(0) {}];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...
}
