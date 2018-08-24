fn main() {
    assert_eq!((ToString::to_string as fn(&(ToString+'static)) -> String)(&"foo"),
        String::from("foo"));
}
