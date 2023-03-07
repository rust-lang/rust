fn main() {
    let x = (0, 2);

    match x {
        (0, ref y) | (y, 0) => {} //~ ERROR E0409
                                  //~| ERROR E0308
        _ => ()
    }
}
