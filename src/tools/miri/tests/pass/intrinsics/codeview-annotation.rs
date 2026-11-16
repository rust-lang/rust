// Verifies that `codeview_annotation` works under Miri

#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

use std::intrinsics::{CodeViewAnnotationArgs, codeview_annotation};

struct Args;

impl CodeViewAnnotationArgs for Args {
    const ARGS: &[&str] = &["Hello", "World"];
}

fn main() {
    codeview_annotation::<Args>();
}
