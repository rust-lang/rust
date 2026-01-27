//~ ERROR: cycle detected when looking up span for `Default`
trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    format!("{}", 0);
}
