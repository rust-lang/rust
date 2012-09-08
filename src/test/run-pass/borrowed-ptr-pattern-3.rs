fn foo<T>(s: &r/uint) {
    match s {
        &3 => fail ~"oh",
        _ => ()
    }
}

fn main() {

}