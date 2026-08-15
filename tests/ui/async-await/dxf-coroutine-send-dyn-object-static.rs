//@ revisions: without_aob aob_only with_aob
//@ edition: 2024
//@ [without_aob] compile-flags: -Znext-solver -Zdxf
//@ [without_aob] known-bug: #149235
//@ [aob_only] compile-flags: -Znext-solver -Zassumptions-on-binders
//@ [aob_only] check-pass
//@ [with_aob] compile-flags: -Znext-solver -Zdxf -Zassumptions-on-binders
//@ [with_aob] check-pass

// Minimized from issue #149235.
//
// `Wrapper<dyn Any>` has `unsafe impl<C: ObjectMarker + ?Sized> Send for Wrapper<C>`
// and `impl ObjectMarker for dyn Any` (implicitly `dyn Any + 'static`). After MIR
// region erasure, `'static` on the `dyn Any` becomes `ReErased`.
// `coroutine_hidden_types` rebinds it as a universally-quantified `BoundVar`.
// The solver must prove `Wrapper<dyn Any + '!0>: Send` for placeholder `'!0`,
// which requires `dyn Any + '!0: ObjectMarker`. But `ObjectMarker` is only
// implemented for `dyn Any + 'static`, not for arbitrary `'!0`.
//
// AoB seems to resolve it by accident because it might be discharging an outlive
// obligation.

#![allow(dead_code, private_bounds)]

use std::any::Any;
use std::future::Future;
use std::marker::PhantomData;

pub struct HasDropImpl;

impl Drop for HasDropImpl {
    fn drop(&mut self) {}
}

pub struct Wrapper<Context: ObjectMarker + ?Sized = dyn Any> {
    raw: HasDropImpl,
    _param: PhantomData<Context>,
}

unsafe impl<C: ObjectMarker + ?Sized> Send for Wrapper<C> {}

trait ObjectMarker {}

impl ObjectMarker for dyn Any {}

fn fails() -> Result<(), Wrapper> {
    Ok(())
}

async fn fut1() {}

fn assert_send<F: Future + Send>(_: F) {}

fn main() {
    let fut = async {
        let _across_await = fails();
        fut1().await;
    };
    assert_send(fut);
}
