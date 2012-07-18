// error-pattern:failed to resolve imports
// n.b. Can't use a //~ ERROR because there's a non-spanned error
// message.
import x = m::f;

mod m {
}

fn main() {
}