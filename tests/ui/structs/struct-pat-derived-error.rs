struct A {
    b: usize,
    c: usize
}

impl A {
    fn foo(&self) {
        let A { x, y } = self.d; //~ ERROR no field `d` on type `&A`
        //~^ ERROR struct `A` does not have fields named `x`, `y`
        //~| ERROR pattern does not mention fields `b`, `c`
    }
}

fn main() {}
