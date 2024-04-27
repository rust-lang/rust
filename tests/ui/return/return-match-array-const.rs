fn main() {
//~^ NOTE: not the enclosing function body
//~| NOTE: not the enclosing function body
//~| NOTE: not the enclosing function body
    [(); return match 0 { n => n }];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...

    [(); return match 0 { 0 => 0 }];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...

    [(); return match () { 'a' => 0, _ => 0 }];
    //~^ ERROR: return statement outside of function body [E0572]
    //~| NOTE: the return is part of this body...
    //~| ERROR: mismatched types [E0308]
    //~| NOTE: expected `()`, found `char`
    //~| NOTE: this expression has type `()`
}
