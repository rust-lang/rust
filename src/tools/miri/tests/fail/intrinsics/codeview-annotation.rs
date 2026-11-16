// Verifies that `codeview_annotation` produces errors under Miri
// when args fail to evaluate

#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

use std::intrinsics::{CodeViewAnnotationArgs, codeview_annotation};

struct Args;

impl CodeViewAnnotationArgs for Args {
    const ARGS: &[&str] = panic!("panic"); //~ ERROR: evaluation panicked
}

fn main() {
    codeview_annotation::<Args>();
}
