trait T {
    fn f(&self, _: ()) {
        None::<()>.map(Self::f);
    }
    //~^^ ERROR function is expected to take a single 0-tuple as argument
}

fn main() {}
