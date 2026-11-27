fn main() {
    fn f() -> Option<i32> {
        Some("fffffffsssssssssddddssssfffffddddff")
            .map(|s| s)
            .map(|s| s.to_string())
            .map(|res| match Some(res) {
                Some(ref s) if s == "" => 41,
                Some(_) => 42,
                _ => 43,
            })
    }
    println!("{:?}", f())
}
