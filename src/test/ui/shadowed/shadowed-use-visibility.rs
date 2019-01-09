mod foo {
    pub fn f() {}

    use foo as bar;
    pub use self::f as bar;
}

mod bar {
    use foo::bar::f as g; //~ ERROR module `bar` is private

    use foo as f;
    pub use foo::*;
}

use bar::f::f; //~ ERROR module `f` is private
fn main() {}
