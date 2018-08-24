trait Trait { }

fn function(t: &mut Trait) {
    t as *mut Trait
 //~^ ERROR: mismatched types
}

fn main() { }
