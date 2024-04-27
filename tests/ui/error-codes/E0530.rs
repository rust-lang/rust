fn main() {
    static TEST: i32 = 0;

    let r: (i32, i32) = (0, 0);
    match r {
        TEST => {} //~ ERROR E0530
    }
}
