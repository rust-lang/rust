// xfail-fast
// aux-build:static-methods-crate.rs

use static_methods_crate;
import static_methods_crate::read;
import readMaybeRenamed = static_methods_crate::readMaybe;

fn main() {
    assert read(~"5") == 5;
    assert readMaybeRenamed(~"false") == Some(false);
    assert readMaybeRenamed(~"foo") == None::<bool>;
}
