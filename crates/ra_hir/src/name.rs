use std::fmt;

use ra_syntax::{ast, SmolStr};

/// `Name` is a wrapper around string, which is used in hir for both references
/// and declarations. In theory, names should also carry hygene info, but we are
/// not there yet!
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name {
    text: SmolStr,
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl Name {
    // TODO: get rid of this?
    pub(crate) fn as_str(&self) -> &str {
        self.text.as_str()
    }

    #[cfg(not(test))]
    fn new(text: SmolStr) -> Name {
        Name { text }
    }

    #[cfg(test)]
    pub(crate) fn new(text: SmolStr) -> Name {
        Name { text }
    }
}

pub(crate) trait AsName {
    fn as_name(&self) -> Name;
}

impl AsName for ast::NameRef<'_> {
    fn as_name(&self) -> Name {
        Name::new(self.text())
    }
}

impl AsName for ast::Name<'_> {
    fn as_name(&self) -> Name {
        Name::new(self.text())
    }
}

impl AsName for ra_db::Dependency {
    fn as_name(&self) -> Name {
        Name::new(self.name.clone())
    }
}
