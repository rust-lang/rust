// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

union U {
    principal: u8,
}

impl U {
    fn calculate(&self) {}
}

fn main() {
    let u = U { principle: 0 };
    //~^ ERROR union `U` has no field named `principle`
    //~| HELP did you mean `principal`?
    //~| SUGGESTION principal
    let w = u.principial; //~ ERROR no field `principial` on type `U`
                          //~| HELP did you mean `principal`?
                          //~| SUGGESTION principal

    let y = u.calculate; //~ ERROR attempted to take value of method `calculate` on type `U`
                         //~| HELP use parentheses to call the method
                         //~| SUGGESTION ()
}
