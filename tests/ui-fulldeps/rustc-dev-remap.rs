//@ check-fail
//
//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//
//@ revisions: only-remap remap-unremap
//@ compile-flags: -Z simulate-remapped-rust-src-base=/rustc-dev/xyz
//@ [remap-unremap]compile-flags: -Ztranslate-remapped-path-to-local-path=yes

// The $SRC_DIR*.rs:LL:COL normalisation doesn't kick in automatically
// as the remapped revision will begin with $COMPILER_DIR_REAL,
// so we have to do it ourselves.
//@ normalize-stderr: ".rs:\d+:\d+" -> ".rs:LL:COL"

#![feature(rustc_private)]

extern crate rustc_ast;

use rustc_ast::visit::Visitor;

struct MyStruct;
struct NotAValidResultType;

impl Visitor<'_> for MyStruct {
    type Result = NotAValidResultType;
    //~^ ERROR the trait bound `NotAValidResultType: VisitorResult` is not satisfied
}

fn main() {}
