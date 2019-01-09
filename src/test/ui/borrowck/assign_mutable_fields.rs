// Currently, we permit you to assign to individual fields of a mut
// var, but we do not permit you to use the complete var afterwards.
// We hope to fix this at some point.
//
// FIXME(#54987)

fn assign_both_fields_and_use() {
    let mut x: (u32, u32);
    x.0 = 1;
    x.1 = 22;
    drop(x.0); //~ ERROR
    drop(x.1); //~ ERROR
}

fn assign_both_fields_the_use_var() {
    let mut x: (u32, u32);
    x.0 = 1;
    x.1 = 22;
    drop(x); //~ ERROR
}

fn main() { }
