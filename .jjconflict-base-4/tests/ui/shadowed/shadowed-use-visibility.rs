mod foo {
    pub fn f() {}

    use crate::foo as bar;
    pub use self::f as bar;
}

mod bar {
    use crate::foo::bar::f as g; //~ ERROR module import `bar` is private

    use crate::foo as f;
    pub use crate::foo::*;
}

use bar::f::f; //~ ERROR module import `f` is private
fn main() {}
