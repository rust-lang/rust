fn main() {
    let tuple = (((),),);

    let _ = tuple. 0.0; //~ ERROR unexpected token: `0.0`

    let _ = tuple./*special cases*/0.0; //~ ERROR unexpected token: `0.0`
}
