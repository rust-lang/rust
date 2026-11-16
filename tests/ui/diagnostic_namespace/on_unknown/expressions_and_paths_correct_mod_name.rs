#![feature(diagnostic_on_unknown)]

mod a {
    #[diagnostic::on_unknown(
        message = "this={this}, unresolved={unresolved}",
    )]
    pub mod b {}
}

mod c {
    pub use super::a::b as d;
}

use c::d::x; //~ ERROR this=b, unresolved=x

fn main() {
    let _ = c::d::X(); //~ ERROR this=b, unresolved=X

    let _: c::d::X; //~ ERROR this=b, unresolved=X
}
