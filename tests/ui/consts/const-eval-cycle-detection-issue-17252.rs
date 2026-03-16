// https://github.com/rust-lang/rust/issues/17252
// Tests that constant evaluation cycles (self-referential consts) are detected
// and reported as errors instead of causing a stack overflow.
const FOO: usize = FOO; //~ ERROR E0391
//@ ignore-parallel-frontend query cycle
fn main() {
    let _x: [u8; FOO]; // caused stack overflow prior to fix
    let _y: usize = 1 + {
        const BAR: usize = BAR;
        //~^ ERROR: cycle
        let _z: [u8; BAR]; // caused stack overflow prior to fix
        1
    };
}
