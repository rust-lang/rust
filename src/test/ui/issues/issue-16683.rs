trait T<'a> {
    fn a(&'a self) -> &'a bool;
    fn b(&self) {
        self.a(); //~ ERROR cannot infer
    }
}

fn main() {}
