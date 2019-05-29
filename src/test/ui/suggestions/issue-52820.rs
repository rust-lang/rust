struct Bravery {
    guts: String,
    brains: String,
}

fn main() {
    let guts = "mettle";
    let _ = Bravery {
        guts, //~ ERROR mismatched types
        brains: guts.clone(), //~ ERROR mismatched types
    };
}
