// Test for -Z llvm_module_flags
//@ compile-flags: -Z llvm_module_flag=foo:u32:123:error -Z llvm_module_flag=bar:u32:42:max

fn main() {}

// CHECK: !{i32 1, !"foo", i32 123}
// CHECK: !{i32 7, !"bar", i32 42}
