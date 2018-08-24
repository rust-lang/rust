// compile-flags: -F deprecated

#[allow(deprecated)] //~ ERROR allow(deprecated) overruled by outer forbid(deprecated)
fn main() {
}
