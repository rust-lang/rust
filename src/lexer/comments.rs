use lexer::ptr::Ptr;

use {SyntaxKind};
use syntax_kinds::*;

pub(crate) fn scan_shebang(ptr: &mut Ptr) -> bool {
    if ptr.next_is('!') && ptr.nnext_is('/') {
        ptr.bump();
        ptr.bump();
        bump_until_eol(ptr);
        true
    } else {
        false
    }
}

pub(crate) fn scan_comment(ptr: &mut Ptr) -> Option<SyntaxKind> {
    if ptr.next_is('/') {
        bump_until_eol(ptr);
        Some(COMMENT)
    } else {
        None
    }
}


fn bump_until_eol(ptr: &mut Ptr) {
    loop {
        if ptr.next_is('\n') || ptr.next_is('\r') && ptr.nnext_is('\n') {
            return;
        }
        if ptr.bump().is_none() {
            break;
        }
    }
}