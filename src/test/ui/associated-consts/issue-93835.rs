fn e() {
    p:a<p:p<e=6>>
    //~^ ERROR comparison operators
    //~| ERROR cannot find value
    //~| ERROR associated const equality
    //~| ERROR associated const equality
    //~| ERROR associated type bounds
}

fn main() {}
