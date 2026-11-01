fn main() {
    let y = Some(0);
    if let Some(x) = y {
        x = 2; //~ ERROR cannot assign twice to immutable variable `x`
    }

    let mut arr = [1, 2, 3];
    let [x, ref xs_hold @ ..] = arr;
    x = 0; //~ ERROR cannot assign twice to immutable variable `x`
    eprintln!("{:?}", arr);
}
