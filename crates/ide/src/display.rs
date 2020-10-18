//! This module contains utilities for turning SyntaxNodes and HIR types
//! into types that may be used to render in a UI.

mod navigation_target;
mod short_label;

pub use navigation_target::NavigationTarget;
pub(crate) use navigation_target::{ToNav, TryToNav};
pub(crate) use short_label::ShortLabel;

pub(crate) use syntax::display::{function_declaration, macro_label};
