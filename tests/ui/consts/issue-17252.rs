const FOO: usize = FOO; //~ ERROR E0391

fn main() {
    let _x: [u8; FOO]; // caused stack overflow prior to fix
    let _y: usize = 1 + {
        const BAR: usize = BAR;
        //~^ ERROR: cycle
        let _z: [u8; BAR]; // caused stack overflow prior to fix
        1
    };
}
