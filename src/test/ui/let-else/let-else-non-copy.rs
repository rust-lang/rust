// run-pass
//
// This is derived from a change to compiler/rustc_codegen_llvm/src/debuginfo/metadata.rs, in
// preparation for adopting let-else within the compiler (thanks @est31):
//
// ```
// -    let place = if let mir::VarDebugInfoContents::Place(p) = var.value { p } else { continue };
// +    let mir::VarDebugInfoContents::Place(place) = var.value else { continue };
// ```
//
// The move was due to mir::Place being Copy, but mir::VarDebugInfoContents not being Copy.

#![feature(let_else)]

#[derive(Copy, Clone)]
struct Copyable;

enum NonCopy {
    Thing(Copyable),
    #[allow(unused)]
    Other,
}

struct Wrapper {
    field: NonCopy,
}

fn let_else() {
    let vec = vec![Wrapper { field: NonCopy::Thing(Copyable) }];
    for item in &vec {
        let NonCopy::Thing(_copyable) = item.field else { continue };
    }
}

fn if_let() {
    let vec = vec![Wrapper { field: NonCopy::Thing(Copyable) }];
    for item in &vec {
        let _copyable = if let NonCopy::Thing(copyable) = item.field { copyable } else { continue };
    }
}

fn main() {
    let_else();
    if_let();
}
