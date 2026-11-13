pub trait TraitA { fn method_a(&self) {} }
pub trait TraitB { fn method_b(&self) {} }
pub mod prelude {
    pub use crate::TraitA as _;
    pub use crate::TraitB as _;
}
