fn main() {
    let error = error; //~ ERROR cannot find value `error`

    // These used to cause errors.
    0 as f32;
    0.0 as u32;
}
