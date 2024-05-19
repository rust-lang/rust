extern "C" {
    fn lol() { //~ ERROR incorrect function inside `extern` block
        println!("");
    }
}
fn main() {}
