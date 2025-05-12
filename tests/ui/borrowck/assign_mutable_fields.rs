// Currently, we do permit you to assign to individual fields of an
// uninitialized var.
// We hope to fix this at some point.
//
// FIXME(#54987)

fn assign_both_fields_and_use() {
    let mut x: (u32, u32);
    x.0 = 1; //~ ERROR
    x.1 = 22;
    drop(x.0);
    drop(x.1);
}

fn assign_both_fields_the_use_var() {
    let mut x: (u32, u32);
    x.0 = 1; //~ ERROR
    x.1 = 22;
    drop(x);
}

fn main() { }
