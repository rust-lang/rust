// xfail-fast
// aux-build:static-methods-crate.rs
#[legacy_exports];

extern mod static_methods_crate;
use static_methods_crate::read;
use readMaybeRenamed = static_methods_crate::readMaybe;

fn main() {
    let result: int = read(~"5");
    assert result == 5;
    assert readMaybeRenamed(~"false") == Some(false);
    assert readMaybeRenamed(~"foo") == None::<bool>;
}
