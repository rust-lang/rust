fn main() {
    // Make sure match uses the usual pointer comparison code path -- i.e., it should complain
    // that pointer comparison is disallowed, not that parts of a pointer are accessed as raw
    // bytes.
    let _: [u8; 0] = [4; { //~ ERROR could not evaluate repeat length
        match &1 as *const i32 as usize { //~ ERROR raw pointers cannot be cast to integers
            0 => 42, //~ ERROR constant contains unimplemented expression type
            //~^ NOTE "pointer arithmetic or comparison" needs an rfc before being allowed
            n => n,
        }
    }];
}
