#[allow(dead_code)]
enum Two {
    A,
    B,
}
impl Drop for Two {
    fn drop(&mut self) {}
}
fn main() {
    let _k = Two::A;
}
