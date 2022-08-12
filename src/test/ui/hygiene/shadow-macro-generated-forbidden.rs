// Check that macro-generated statics and consts are not allowed to be shadowed

macro_rules! h {
    () => {
        #[allow(non_upper_case_globals)]
        static x: usize = 2;
        #[allow(non_upper_case_globals)]
        const y: usize = 3;
    }
}

h!();

fn main() {
    let x @ _ = 4;
    //~^ ERROR let bindings cannot shadow statics
    let y @ _ = 5;
    //~^ ERROR let bindings cannot shadow constants
}
