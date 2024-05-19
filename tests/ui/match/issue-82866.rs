fn main() {
    match x {
        //~^ ERROR cannot find value `x` in this scope
        Some::<v>(v) => (),
        //~^ ERROR cannot find type `v` in this scope
    }
}
