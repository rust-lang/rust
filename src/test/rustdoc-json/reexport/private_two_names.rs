// Test for the ICE in rust/83720
// A pub-in-private type re-exported under two different names shouldn't cause an error

mod style {
    pub struct Color;
}

pub use style::Color;
pub use style::Color as Colour;
