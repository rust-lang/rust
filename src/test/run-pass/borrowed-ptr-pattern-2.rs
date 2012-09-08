fn foo<T>(s: &str) {
    match s {
        &"kitty" => fail ~"cat",
        _ => ()
    }
}

fn main() {

}