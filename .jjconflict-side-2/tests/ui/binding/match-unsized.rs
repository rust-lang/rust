//@ run-pass
fn main() {
    let data: &'static str = "Hello, World!";
    match data {
        &ref xs => {
            assert_eq!(data, xs);
        }
    }
}
