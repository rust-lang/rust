
fn something(f: pure fn()) { f(); }
fn main() {
    something(|| log(error, "hi!") );
}
