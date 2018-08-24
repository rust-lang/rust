union U {
    principal: u8,
}

impl U {
    fn calculate(&self) {}
}

fn main() {
    let u = U { principle: 0 };
    //~^ ERROR union `U` has no field named `principle`
    let w = u.principial; //~ ERROR no field `principial` on type `U`
                          //~^ did you mean `principal`?

    let y = u.calculate; //~ ERROR attempted to take value of method `calculate` on type `U`
}
