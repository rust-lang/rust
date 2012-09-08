fn foo<T>(x: &T) {
    match x {
        &a => fail #fmt("%?", a)
    }
}

fn main() {

}