struct a {
    b: usize,
    c: usize
}

impl a {
    fn foo(&self) {
        let a { x, y } = self.d; //~ ERROR no field `d` on type `&a`
        //~^ ERROR struct `a` does not have fields named `x`, `y`
        //~| ERROR pattern does not mention fields `b`, `c`
    }
}

fn main() {}
