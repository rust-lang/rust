//! Builds upon [`crate::ast::make`] constructors to create ast fragments with
//! optional syntax mappings.
//!
//! Instead of forcing make constructors to perform syntax mapping, we instead
//! let [`SyntaxFactory`] handle constructing the mappings. Care must be taken
//! to remember to feed the syntax mappings into a [`SyntaxEditor`](crate::syntax_editor::SyntaxEditor),
//! if applicable.

mod constructors;

use std::cell::{RefCell, RefMut};

use crate::syntax_editor::SyntaxMapping;

#[derive(Debug)]
pub struct SyntaxFactory {
    // Stored in a refcell so that the factory methods can be &self
    mappings: Option<RefCell<SyntaxMapping>>,
}

impl SyntaxFactory {
    /// Creates a new [`SyntaxFactory`], generating mappings between input nodes and generated nodes.
    pub(crate) fn with_mappings() -> Self {
        Self { mappings: Some(RefCell::new(SyntaxMapping::default())) }
    }

    /// Creates a [`SyntaxFactory`] without generating mappings.
    pub fn without_mappings() -> Self {
        Self { mappings: None }
    }

    /// Take all of the tracked syntax mappings, leaving `SyntaxMapping::default()` in its place, if any.
    pub(crate) fn take(&self) -> SyntaxMapping {
        self.mappings.as_ref().map(|mappings| mappings.take()).unwrap_or_default()
    }

    pub(crate) fn mappings(&self) -> Option<RefMut<'_, SyntaxMapping>> {
        self.mappings.as_ref().map(|it| it.borrow_mut())
    }
}

impl Default for SyntaxFactory {
    fn default() -> Self {
        Self::without_mappings()
    }
}
