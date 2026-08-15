// Regression test for issue #159482
// Misspelled compiler options should suggest the closest valid command-line spelling.

//@ revisions: codegen unstable
//@[codegen] compile-flags: -Copt-leve=2
//@[unstable] compile-flags: -Zno-analysi

fn main() {}

//[codegen]~? ERROR unknown codegen option: `opt-leve`
//[unstable]~? ERROR unknown unstable option: `no-analysi`
