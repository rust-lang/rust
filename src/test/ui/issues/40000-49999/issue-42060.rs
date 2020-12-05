fn main() {
    let thing = ();
    let other: typeof(thing) = thing; //~ ERROR attempt to use a non-constant value in a constant
    //~^ ERROR `typeof` is a reserved keyword but unimplemented [E0516]
}

fn f(){
    let q = 1;
    <typeof(q)>::N //~ ERROR attempt to use a non-constant value in a constant
    //~^ ERROR `typeof` is a reserved keyword but unimplemented [E0516]
}
