// build-pass

// Check that a macro let binding that is unambiguous is allowed
// to shadow externally defined constants

macro_rules! h {
    () => {
        let x @ _ = 2;
        let y @ _ = 3;
    }
}

#[allow(non_upper_case_globals)]
const x: usize = 4;

#[allow(non_upper_case_globals)]
const y: usize = 5;

#[allow(non_upper_case_globals)]
fn a<const x: usize>() {
    h!();
}

fn main() {
    h!();
}
