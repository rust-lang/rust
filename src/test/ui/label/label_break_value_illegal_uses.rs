#![feature(label_break_value)]

// These are forbidden occurrences of label-break-value

fn labeled_unsafe() {
    unsafe 'b: {} //~ ERROR expected one of `extern`, `fn`, or `{`
}

fn labeled_if() {
    if true 'b: {} //~ ERROR expected `{`, found `'b`
}

fn labeled_else() {
    if true {} else 'b: {} //~ ERROR expected `{`, found `'b`
}

fn labeled_match() {
    match false 'b: {} //~ ERROR expected one of `.`, `?`, `{`, or an operator
}

pub fn main() {}
