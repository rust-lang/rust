//@ run-pass
//@ proc-macro: ver-cfg-rel.rs
//@ revisions: assume no_assume
//@ [assume]compile-flags: -Z assume-incomplete-release
//@ ignore-backends: gcc

#![feature(cfg_version)]

extern crate ver_cfg_rel;

use ver_cfg_rel::ver_cfg_rel;

#[ver_cfg_rel("-2")]
fn foo_2() { }

#[ver_cfg_rel("-1")]
fn foo_1() { }

#[cfg(assume)]
#[ver_cfg_rel("0")]
fn foo() { compile_error!("wrong+0") }

#[cfg(no_assume)]
#[ver_cfg_rel("0")]
fn foo() { }

#[ver_cfg_rel("1")]
fn bar() { compile_error!("wrong+1") }

#[ver_cfg_rel("2")]
fn bar() { compile_error!("wrong+2") }

fn main() {
    foo_2();
    foo_1();

    #[cfg(no_assume)]
    foo();
}
