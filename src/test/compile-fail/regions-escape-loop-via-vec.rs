// The type of `y` ends up getting inferred to the type of the block.
// This generates a ton of error msgs at the moment.
fn broken() -> int {
    let mut x = 3;
    let mut y = ~[&mut x]; //~ ERROR reference is not valid
    while x < 10 {
        let mut z = x;
        y += ~[&mut z];
        x += 1;
    }
    vec::foldl(0, y, |v, p| v + *p )
    //~^ ERROR reference is not valid
    //~^^ ERROR reference is not valid
    //~^^^ ERROR reference is not valid
}

fn main() { }