fn main() {
    let nil = ();
    let _t = nil as usize; //~ ERROR: non-primitive cast: `()` as `usize`
}
