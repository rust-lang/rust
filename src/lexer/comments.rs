use lexer::ptr::Ptr;

use {SyntaxKind};

pub(crate) fn scan_shebang(ptr: &mut Ptr) -> bool {
    false
}

pub(crate) fn scan_comment(ptr: &mut Ptr) -> Option<SyntaxKind> {
    None
}