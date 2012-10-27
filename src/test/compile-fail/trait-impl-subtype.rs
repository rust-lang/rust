trait Mumbo {
    fn jumbo(&self, x: @uint) -> uint;
}

impl uint: Mumbo {
    // Note: this method def is ok, it is more accepting and
    // less effecting than the trait method:
    pure fn jumbo(&self, x: @const uint) -> uint { *self + *x }
}

fn main() {
    let a = 3u;
    let b = a.jumbo(@mut 6);

    let x = @a as @Mumbo;
    let y = x.jumbo(@mut 6); //~ ERROR values differ in mutability
    let z = x.jumbo(@6);
}



