// run-pass

fn main() {
    assert_eq!((ToString::to_string as fn(&(dyn ToString+'static)) -> String)(&"foo"),
        String::from("foo"));
}
