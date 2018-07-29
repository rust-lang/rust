use lexer::ptr::Ptr;
use lexer::classes::*;

use SyntaxKind::{self, *};

pub(crate) fn scan_number(c: char, ptr: &mut Ptr) -> SyntaxKind {
    if c == '0' {
        match ptr.next().unwrap_or('\0') {
            'b' | 'o' => {
                ptr.bump();
                scan_digits(ptr, false);
            }
            'x' => {
                ptr.bump();
                scan_digits(ptr, true);
            }
            '0'...'9' | '_' | '.' | 'e' | 'E' => {
                scan_digits(ptr, true);
            }
            _ => return INT_NUMBER,
        }
    } else {
        scan_digits(ptr, false);
    }

    // might be a float, but don't be greedy if this is actually an
    // integer literal followed by field/method access or a range pattern
    // (`0..2` and `12.foo()`)
    if ptr.next_is('.') && !(ptr.nnext_is('.') || ptr.nnext_is_p(is_ident_start)) {
        // might have stuff after the ., and if it does, it needs to start
        // with a number
        ptr.bump();
        scan_digits(ptr, false);
        scan_float_exponent(ptr);
        return FLOAT_NUMBER;
    }
    // it might be a float if it has an exponent
    if ptr.next_is('e') || ptr.next_is('E') {
        scan_float_exponent(ptr);
        return FLOAT_NUMBER;
    }
    INT_NUMBER
}

fn scan_digits(ptr: &mut Ptr, allow_hex: bool) {
    while let Some(c) = ptr.next() {
        match c {
            '_' | '0'...'9' => {
                ptr.bump();
            }
            'a'...'f' | 'A'...'F' if allow_hex => {
                ptr.bump();
            }
            _ => return,
        }
    }
}

fn scan_float_exponent(ptr: &mut Ptr) {
    if ptr.next_is('e') || ptr.next_is('E') {
        ptr.bump();
        if ptr.next_is('-') || ptr.next_is('+') {
            ptr.bump();
        }
        scan_digits(ptr, false);
    }
}
