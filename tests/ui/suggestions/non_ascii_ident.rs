fn main() {
    // There shall be no suggestions here. In particular not `Ok`.
    let _ = 读文; //~ ERROR cannot find value `读文` in this scope

    let f = 0f32; // Important line to make this an ICE regression test
    读文(f); //~ ERROR cannot find function `读文` in this scope
}
