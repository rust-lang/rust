use SyntaxKind::{self, *};

use lexer::ptr::Ptr;

pub(crate) fn is_string_literal_start(c: char, c1: Option<char>, c2: Option<char>) -> bool {
    match (c, c1, c2) {
        ('r', Some('"'), _)
        | ('r', Some('#'), _)
        | ('b', Some('"'), _)
        | ('b', Some('\''), _)
        | ('b', Some('r'), Some('"'))
        | ('b', Some('r'), Some('#')) => true,
        _ => false,
    }
}

pub(crate) fn scan_char(ptr: &mut Ptr) {
    if ptr.bump().is_none() {
        return; // TODO: error reporting is upper in the stack
    }
    scan_char_or_byte(ptr);
    if !ptr.next_is('\'') {
        return; // TODO: error reporting
    }
    ptr.bump();
}

pub(crate) fn scan_byte_char_or_string(ptr: &mut Ptr) -> SyntaxKind {
    // unwrapping and not-exhaustive match are ok
    // because of string_literal_start
    let c = ptr.bump().unwrap();
    match c {
        '\'' => {
            scan_byte(ptr);
            BYTE
        }
        '"' => {
            scan_byte_string(ptr);
            BYTE_STRING
        }
        'r' => {
            scan_raw_byte_string(ptr);
            RAW_BYTE_STRING
        }
        _ => unreachable!(),
    }
}

pub(crate) fn scan_string(ptr: &mut Ptr) {
    while let Some(c) = ptr.bump() {
        if c == '"' {
            return;
        }
    }
}

pub(crate) fn scan_raw_string(ptr: &mut Ptr) {
    if !ptr.next_is('"') {
        return;
    }
    ptr.bump();

    while let Some(c) = ptr.bump() {
        if c == '"' {
            return;
        }
    }
}

fn scan_byte(ptr: &mut Ptr) {
    if ptr.next_is('\'') {
        ptr.bump();
        return;
    }
    ptr.bump();
    if ptr.next_is('\'') {
        ptr.bump();
        return;
    }
}

fn scan_byte_string(ptr: &mut Ptr) {
    while let Some(c) = ptr.bump() {
        if c == '"' {
            return;
        }
    }
}

fn scan_raw_byte_string(ptr: &mut Ptr) {
    if !ptr.next_is('"') {
        return;
    }
    ptr.bump();

    while let Some(c) = ptr.bump() {
        if c == '"' {
            return;
        }
    }
}

fn scan_char_or_byte(ptr: &mut Ptr) {
    //FIXME: deal with escape sequencies
    ptr.bump();
}
