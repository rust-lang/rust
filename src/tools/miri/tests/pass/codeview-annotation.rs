// Verifies that `codeview_annotation` is a no-op under Miri
// and that Miri does not try to const eval `CodeViewAnnotationArgs::ARGS`

#![feature(codeview_annotation)]

use std::hint::{CodeViewAnnotationArgs, codeview_annotation};

struct Args;

impl CodeViewAnnotationArgs for Args {
    const ARGS: &[&str] = panic!("Panic triggered if Miri tries to evaluate annotation args");
}

fn main() {
    codeview_annotation::<Args>();
}
