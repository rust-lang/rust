extern "C" {
    static mut a: i32;
}

fn main() {
    a += 3; //~ ERROR: requires unsafe
    //~^ WARN creating a mutable reference to mutable static is discouraged [static_mut_refs]
    a = 4; //~ ERROR: requires unsafe
    let _b = a; //~ ERROR: requires unsafe
}
