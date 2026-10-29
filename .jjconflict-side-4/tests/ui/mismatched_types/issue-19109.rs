trait Trait { }

fn function(t: &mut dyn Trait) {
    t as *mut dyn Trait
 //~^ ERROR: mismatched types
}

fn main() { }
