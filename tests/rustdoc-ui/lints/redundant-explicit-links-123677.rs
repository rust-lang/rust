//@ check-pass
#![deny(rustdoc::redundant_explicit_links)]

mod bar {
    /// [`Rc`](std::rc::Rc)
    pub enum Baz {}
}

pub use bar::*;

use std::rc::Rc;

/// [`Rc::allocator`] [foo](std::rc::Rc)
pub fn winit_runner() {}
