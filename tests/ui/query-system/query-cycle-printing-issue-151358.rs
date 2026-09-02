//~ ERROR: cycle detected
trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    format!("{}", 0);
}
