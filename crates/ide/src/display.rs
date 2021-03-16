//! This module contains utilities for turning SyntaxNodes and HIR types
//! into types that may be used to render in a UI.

pub(crate) mod navigation_target;

pub(crate) use navigation_target::{ToNav, TryToNav};

pub(crate) use syntax::display::macro_label;
