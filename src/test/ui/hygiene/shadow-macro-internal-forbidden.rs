// Test that shadowing of statics and constants from within the same macro is still forbidden.

macro_rules! h {
    () => {
        #[allow(non_upper_case_globals)]
        static x: usize = 2;
        #[allow(non_upper_case_globals)]
        const y: usize = 3;
        fn a() {
            let x @ _ = 4;
            //~^ ERROR let bindings cannot shadow statics
            let y @ _ = 5;
            //~^ ERROR let bindings cannot shadow constants
        }
        #[allow(non_upper_case_globals)]
        fn b<const z: usize>() {
            let z @ _ = 21;
            //~^ ERROR let bindings cannot shadow const parameters
        }
    }
}

h!();

fn main() {}
