fn main() {
    match 10 {
        <T as Trait>::Type{key: value} => (),
        //~^ ERROR unexpected `{` after qualified path
        _ => (),
    }
}
