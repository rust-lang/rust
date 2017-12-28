use text::{TextUnit};
use syntax_kinds::syntax_info;

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxKind(pub(crate) u32);

impl SyntaxKind {
    fn info(self) -> &'static SyntaxInfo {
        syntax_info(self)
    }
}

impl fmt::Debug for SyntaxKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = self.info().name;
        f.write_str(name)
    }
}


pub(crate) struct SyntaxInfo {
    pub name: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Token {
    pub kind: SyntaxKind,
    pub len: TextUnit,
}