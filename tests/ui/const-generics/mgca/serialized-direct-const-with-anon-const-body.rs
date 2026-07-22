//@ check-pass
//@ aux-build:anon-const-def-id-on-const-arg-with-anon.rs
//! The DefCollector sometimes generates "fake" DefKind::AnonConst DefIds for AST AnonConsts that
//! are not actually lowered to HIR AnonConsts, and so the DefId is placed on a hir::Node::ConstArg
//! (just to place it *somewhere*). These "fake" DefIds should not be serialized. Previously, the
//! logic to skip serializing them was incorrect (we were still serializing fake DefIds for
//! `ConstArg(ConstArgKind::Anon)` if the directly represented expression contained within it
//! *another*, unrelated, anon const). This test checks that case, a directly-represented
//! fake-anon-const directly containing another anon const.

#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
extern crate anon_const_def_id_on_const_arg_with_anon;
fn main() {
    anon_const_def_id_on_const_arg_with_anon::f();
}
