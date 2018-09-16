use lexer::ptr::Ptr;

use SyntaxKind::{self, *};

pub(crate) fn scan_shebang(ptr: &mut Ptr) -> bool {
    if ptr.at_str("!/") {
        ptr.bump();
        ptr.bump();
        bump_until_eol(ptr);
        true
    } else {
        false
    }
}

fn scan_block_comment(ptr: &mut Ptr) -> Option<SyntaxKind> {
    if ptr.at('*') {
        ptr.bump();
        let mut depth: u32 = 1;
        while depth > 0 {
            if ptr.at_str("*/") {
                depth -= 1;
                ptr.bump();
                ptr.bump();
            } else if ptr.at_str("/*") {
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
    if ptr.at('/') {
        bump_until_eol(ptr);
        Some(COMMENT)
    } else {
        scan_block_comment(ptr)
    }
}

fn bump_until_eol(ptr: &mut Ptr) {
    loop {
        if ptr.at('\n') || ptr.at_str("\r\n") {
            return;
        }
        if ptr.bump().is_none() {
            break;
        }
    }
}
