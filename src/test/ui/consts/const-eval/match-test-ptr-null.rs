fn main() {
    // Make sure match uses the usual pointer comparison code path -- i.e., it should complain
    // that pointer comparison is disallowed, not that parts of a pointer are accessed as raw
    // bytes.
    let _: [u8; 0] = [4; {
        match &1 as *const i32 as usize {
            //~^ ERROR casting pointers to integers in constants
            //~| ERROR constant contains unimplemented expression type
            //~| ERROR evaluation of constant value failed
            0 => 42, //~ ERROR constant contains unimplemented expression type
            n => n,
        }
    }];
}
