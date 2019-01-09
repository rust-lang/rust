fn main() {
    match 10 {
        <T as Trait>::Type(2) => (),
        //~^ ERROR unexpected `(` after qualified path
        _ => (),
    }
}
