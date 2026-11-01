//@ edition:2021
//@ revisions: afn cls afn_cls nofeat
//@[afn_cls] check-pass

#![feature(stmt_expr_attributes)]
#![deny(ungated_async_fn_track_caller)]
#![cfg_attr(any(afn, afn_cls), feature(async_fn_track_caller))]
#![cfg_attr(any(cls, afn_cls), feature(closure_track_caller))]

fn main() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async {};
}

#[track_caller]
//[cls]~^ ERROR `#[track_caller]` on async functions is a no-op
async fn foo() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async {};
}

#[track_caller]
//[cls]~^ ERROR `#[track_caller]` on async functions is a no-op
async fn foo2() {
    let _ = async {
        let _ = #[track_caller]
        //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        async {};
    };
}
