//@ edition: 2024
//@ revisions: no_dxf dxf dxf_aob
//@ [no_dxf] compile-flags: -Znext-solver -Zassumptions-on-binders
//@ [no_dxf] check-pass
//@ [dxf] compile-flags: -Znext-solver -Zdxf
//@ [dxf] known-bug: #149407
//@ [dxf_aob] compile-flags: -Znext-solver -Zdxf -Zassumptions-on-binders
//@ [dxf_aob] check-pass

// Minimized from issue #149407 (nested async closure / synthetic by-move body).
//
// Under `no_dxf` (AoB alone), passes due to Assumptions-on-Binders.
// Under `dxf` alone, fails with E0477 because higher-ranked closure bounds require AoB.
// Under `dxf_aob` (AoB + DXF), passes cleanly with synthetic by-move body NLL SCC feeding.

use std::future::Future;

struct B;
struct C;

fn u(_c: &C, _b: &B) {}

trait T {
    fn a(&self, b: &B) -> impl Future<Output = ()> + Send;
}

struct Timpl;

impl Timpl {
    async fn b<F>(&self, mut f: F)
    where
        F: AsyncFnMut(&mut C),
    {
        let mut c = C;
        f(&mut c).await;
        f(&mut c).await;
    }
}

impl T for Timpl {
    async fn a(&self, b: &B) {
        self.b(async |c| u(c, b)).await;
    }
}

fn main() {}
