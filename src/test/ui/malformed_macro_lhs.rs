macro_rules! my_precioooous {
    t => (1); //~ ERROR invalid macro matcher
}

fn main() {
    my_precioooous!();
}
