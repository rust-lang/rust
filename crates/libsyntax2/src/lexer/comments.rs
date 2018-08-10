use lexer::ptr::Ptr;

use SyntaxKind::{self, *};

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

fn scan_block_comment(ptr: &mut Ptr) -> Option<SyntaxKind> {
    if ptr.next_is('*') {
        ptr.bump();
        let mut depth: u32 = 1;
        while depth > 0 {
            if ptr.next_is('*') && ptr.nnext_is('/') {
                depth -= 1;
                ptr.bump();
                ptr.bump();
            } else if ptr.next_is('/') && ptr.nnext_is('*') {
                depth += 1;
                ptr.bump();
                ptr.bump();
            } else if ptr.bump().is_none() {
                break;
            }
        }
        Some(COMMENT)
    } else {
        None
    }
}

pub(crate) fn scan_comment(ptr: &mut Ptr) -> Option<SyntaxKind> {
    if ptr.next_is('/') {
        bump_until_eol(ptr);
        Some(COMMENT)
    } else {
        scan_block_comment(ptr)
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
