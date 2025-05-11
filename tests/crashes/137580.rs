//@ known-bug: #137580
fn main() {
    println!("%65536$", 1);
}
