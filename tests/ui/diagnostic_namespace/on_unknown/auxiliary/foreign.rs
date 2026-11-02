#![feature(diagnostic_on_unknown)]

mod a {
    #[diagnostic::on_unknown(
        message = "this={this}, unresolved={unresolved}",
    )]
    pub mod b {}
}

pub mod c {
    pub use super::a::b as d;
}
