//~ ERROR: cycle detected when getting the resolver for lowering
trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    format!("{}", 0);
}
