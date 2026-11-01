// The type of `y` ends up getting inferred to the type of the block.
fn broken() {
    let mut x = 3;
    let mut _y = vec![&mut x];
    while x < 10 { //~ ERROR cannot use `x` because it was mutably borrowed
        let mut z = x; //~ ERROR cannot use `x` because it was mutably borrowed
        _y.push(&mut z);
        //~^ ERROR `z` does not live long enough
        x += 1; //~ ERROR cannot use `x` because it was mutably borrowed
    }
}

fn main() { }
