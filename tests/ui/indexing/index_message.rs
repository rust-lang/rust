fn main() {
    let z = (10,);
    let _ = z[0]; //~ ERROR cannot index into a value of type `({integer},)`
}
