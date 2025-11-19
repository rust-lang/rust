struct Z { }

impl Z {
    fn run(&self, z: &mut Z) { }
    fn start(&mut self) {
        self.run(&mut self); //~ ERROR: cannot borrow
        //~| ERROR: cannot borrow
        //~| HELP: if there is only one mutable reborrow, remove the `&mut`
    }
}

fn main() {
    let mut z = Z {};
    z.start();
}
