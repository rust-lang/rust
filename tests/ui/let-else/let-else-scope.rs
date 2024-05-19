fn main() {
    let Some(x) = Some(2) else {
        panic!("{}", x); //~ ERROR cannot find value `x` in this scope
    };
}
