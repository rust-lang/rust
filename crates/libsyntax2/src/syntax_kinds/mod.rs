mod generated;

use std::fmt;
use SyntaxKind::*;

pub use self::generated::SyntaxKind;

impl fmt::Debug for SyntaxKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = self.info().name;
        f.write_str(name)
    }
}

pub(crate) struct SyntaxInfo {
    pub name: &'static str,
}

impl SyntaxKind {
    pub fn is_trivia(self) -> bool {
        match self {
            WHITESPACE | COMMENT | DOC_COMMENT => true,
            _ => false,
        }
    }
}
