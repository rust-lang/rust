fn main() {
    let _ = (((),),).1.0; //~ ERROR no field `1` on type `(((),),)`

    let _ = (((),),).0.1; //~ ERROR no field `1` on type `((),)`
}
