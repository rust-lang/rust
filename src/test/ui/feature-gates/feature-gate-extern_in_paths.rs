struct S;

fn main() {
    let _ = extern::std::vec::Vec::new(); //~ ERROR `extern` in paths is experimental
}
