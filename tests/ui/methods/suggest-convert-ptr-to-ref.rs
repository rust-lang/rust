fn main() {
    let mut x = 8u8;
    let z: *const u8 = &x;
    // issue #21596
    println!("{}", z.to_string()); //~ ERROR E0599

    let t: *mut u8 = &mut x;
    println!("{}", t.to_string()); //~ ERROR E0599
    t.make_ascii_lowercase(); //~ ERROR E0599

    // suggest `as_mut` simply because the name is similar
    let _ = t.as_mut_ref(); //~ ERROR E0599
    let _ = t.as_ref_mut(); //~ ERROR E0599

    // no ptr-to-ref suggestion
    z.make_ascii_lowercase(); //~ ERROR E0599
}
