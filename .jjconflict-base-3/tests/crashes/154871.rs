//@ known-bug: #154871
struct Struct {
    b: unsafe<> (),
}
fn main() {
    std::ptr::null::<Struct>;
}
