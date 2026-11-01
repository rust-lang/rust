//@ only-wasm32
// Check that link_section on wasm emits both a data segment in the given
// section and a custom section.

#![crate_type = "lib"]
#![feature(core_intrinsics, link_llvm_intrinsics)]

#[unsafe(link_section = "foo")]
#[unsafe(no_mangle)]
#[used]
static FOO: [u8; 22] = *b"custom section content";

// Data segment in foo section
// CHECK: @FOO = dso_local constant [22 x i8] c"custom section content", section "foo"
// CHECK: @llvm.used = appending global [1 x ptr] [ptr @FOO], section "llvm.metadata"

// Custom section
// CHECK: !wasm.custom_sections = !{!3}
// CHECK: !3 = !{!"foo", !"custom section content"}
