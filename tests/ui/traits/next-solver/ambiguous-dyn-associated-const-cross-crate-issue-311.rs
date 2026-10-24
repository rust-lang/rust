//@ aux-build: dyn-associated-const-cross-crate-aux-issue-311.rs
//@ revisions: metadata codegen
//@ compile-flags: -Znext-solver
//@[metadata] check-pass
//@[codegen] build-fail
//@ forbid-output: internal compiler error

// Checking metadata does not require evaluating this constant.
// Codegen does, and must report the ambiguity without an ICE.
extern crate dyn_associated_const_cross_crate_aux_issue_311 as aux;

fn main() {
    println!("{}", aux::Adapter::<aux::Conflicting>::SIZE);
}

//[codegen]~? ERROR cannot evaluate associated constant with unresolved type dependencies

//@ normalize-stderr: "\n\n$" -> "\n"
