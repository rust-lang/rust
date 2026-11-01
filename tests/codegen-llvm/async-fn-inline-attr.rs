// Checks that inline attributes on `async fn`s and async closures apply to the coroutine holding
// their body (i.e. the `Future::poll` implementation) rather than to the function or closure that
// merely constructs it. See #129347.
//
//@ edition: 2024
//@ compile-flags: -Cno-prepopulate-passes -Csymbol-mangling-version=v0 -Zinline-mir=no
#![crate_type = "lib"]
#![feature(stmt_expr_attributes)]

use std::future::Future;
use std::pin::pin;
use std::task::Context;

// CHECK:      ; async_fn_inline_attr::never{{$}}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT:  noinline
// CHECK-SAME: {{$}}
// CHECK:      ; async_fn_inline_attr::never::{closure#0}
// CHECK-NEXT: ; Function Attrs: noinline
#[inline(never)]
pub async fn never() {}

// CHECK:      ; async_fn_inline_attr::always{{$}}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT:  alwaysinline
// CHECK-SAME: {{$}}
// CHECK:      ; async_fn_inline_attr::always::{closure#0}
// CHECK-NEXT: ; Function Attrs: alwaysinline
#[inline(always)]
pub async fn always() {}

// The body coroutine would get an inline hint regardless, like all closures.
// CHECK:      ; async_fn_inline_attr::hint{{$}}
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT:  inlinehint
// CHECK-SAME: {{$}}
// CHECK:      ; async_fn_inline_attr::hint::{closure#0}
// CHECK-NEXT: ; Function Attrs: inlinehint
#[inline]
pub async fn hint() {}

pub struct S;

impl S {
    // CHECK:      ; <async_fn_inline_attr::S>::method{{$}}
    // CHECK-NEXT: ; Function Attrs:
    // CHECK-NOT:  noinline
    // CHECK-SAME: {{$}}
    // CHECK:      ; <async_fn_inline_attr::S>::method::{closure#0}
    // CHECK-NEXT: ; Function Attrs: noinline
    #[inline(never)]
    pub async fn method(&self) {}
}

// CHECK:      ; async_fn_inline_attr::poll_all{{$}}
pub fn poll_all(cx: &mut Context<'_>) {
    let _ = pin!(never()).poll(cx);
    let _ = pin!(always()).poll(cx);
    let _ = pin!(hint()).poll(cx);
    let _ = pin!(S.method()).poll(cx);

    // CHECK:      ; async_fn_inline_attr::poll_all::{closure#0}{{$}}
    // CHECK-NEXT: ; Function Attrs:
    // CHECK-NOT:  noinline
    // CHECK-SAME: {{$}}
    // CHECK:      ; async_fn_inline_attr::poll_all::{closure#0}::{closure#0}::<i16>
    // CHECK-NEXT: ; Function Attrs: noinline
    let closure = #[inline(never)]
    async || {};
    let _ = pin!(closure()).poll(cx);
}
