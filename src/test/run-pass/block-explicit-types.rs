pub fn main() {
    fn as_buf<T, F>(s: String, f: F) -> T where F: FnOnce(String) -> T { f(s) }
    as_buf("foo".to_string(), |foo: String| -> () { println!("{}", foo) });
}
