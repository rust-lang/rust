// https://github.com/rust-lang/rust/issues/17252

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
