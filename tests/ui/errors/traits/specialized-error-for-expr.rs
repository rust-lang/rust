fn main() {
    let s = String::from("hello");
    for _ in s {} //~ `String` is not an iterator [E0277]
}
