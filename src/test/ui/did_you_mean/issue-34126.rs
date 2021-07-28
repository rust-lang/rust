struct Z { }

impl Z {
    fn run(&self, z: &mut Z) { }
    fn start(&mut self) {
        self.run(&mut self); //~ ERROR cannot borrow
        //~| ERROR cannot borrow
        //~| HELP try removing `&mut` here
    }
}

fn main() {
    let mut z = Z {};
    z.start();
}
