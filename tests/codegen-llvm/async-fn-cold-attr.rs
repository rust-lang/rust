// Checks that `#[cold]` on `async fn`s and async closures applies to the coroutine holding their
// body (i.e. the `Future::poll` implementation), in addition to the function or closure that
// constructs it.
//
//@ edition: 2024
//@ compile-flags: -Cno-prepopulate-passes -Csymbol-mangling-version=v0 -Zinline-mir=no
#![crate_type = "lib"]
#![feature(stmt_expr_attributes)]

use std::future::Future;
use std::pin::pin;
use std::task::Context;

// CHECK:      ; async_fn_cold_attr::cold{{$}}
// CHECK-NEXT: ; Function Attrs: cold
// CHECK:      ; async_fn_cold_attr::cold::{closure#0}
// CHECK-NEXT: ; Function Attrs: cold
#[cold]
pub async fn cold() {}

// CHECK:      ; async_fn_cold_attr::not_cold::{closure#0}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT:  cold
// CHECK-SAME: {{$}}
pub async fn not_cold() {}

pub struct S;

impl S {
    // CHECK:      ; <async_fn_cold_attr::S>::method{{$}}
    // CHECK-NEXT: ; Function Attrs: cold
    // CHECK:      ; <async_fn_cold_attr::S>::method::{closure#0}
    // CHECK-NEXT: ; Function Attrs: cold
    #[cold]
    pub async fn method(&self) {}
}

// CHECK:      ; async_fn_cold_attr::poll_all{{$}}
pub fn poll_all(cx: &mut Context<'_>) {
    let _ = pin!(cold()).poll(cx);
    let _ = pin!(not_cold()).poll(cx);
    let _ = pin!(S.method()).poll(cx);

    // CHECK:      ; async_fn_cold_attr::poll_all::{closure#0}{{$}}
    // CHECK-NEXT: ; Function Attrs: cold
    // CHECK:      ; async_fn_cold_attr::poll_all::{closure#0}::{closure#0}::<i16>
    // CHECK-NEXT: ; Function Attrs: cold
    let closure = #[cold]
    async || {};
    let _ = pin!(closure()).poll(cx);
}
