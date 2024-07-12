fn main() {
    match x {
        //~^ ERROR cannot find value `x`
        Some::<v>(v) => (),
        //~^ ERROR cannot find type `v`
    }
}
