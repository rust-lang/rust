macro_rules! foo { () => ( x ) }

fn main() {
    let foo!() = 2;
    x + 1; //~ ERROR cannot find value `x` in this scope
}
