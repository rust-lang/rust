//@ edition:2024
//@ revisions: afn cls afn_cls nofeat
//@[afn_cls] check-pass

#![feature(stmt_expr_attributes, coroutines, gen_blocks)]
#![allow(incomplete_features)]
#![deny(ungated_async_fn_track_caller)]
#![cfg_attr(any(afn, afn_cls), feature(async_fn_track_caller))]
#![cfg_attr(any(cls, afn_cls), feature(closure_track_caller))]

fn main() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async || {};
}

#[track_caller]
//[cls]~^ ERROR `#[track_caller]` on async functions is a no-op
async fn foo() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async || {};
}

async fn foo2() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    || {};
}

fn foo3() {
    let _ = async {
        let _ = #[track_caller]
        //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        || {};
    };
}

async fn foo4() {
    let _ = || {
        #[track_caller]
        //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        || {};
    };
}

fn foo5() {
    let _ = async {
        let _ = || {
            #[track_caller]
            //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
            || {};
        };
    };
}

// FIXME(gen_blocks): #[track_caller] is apparently not properly linted here?
#[track_caller]
gen fn foo6() {}

fn foo7() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    gen {
        yield ();
    };
}

fn foo8() {
    let _ = #[track_caller]
    //[nofeat,afn]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    #[coroutine]
    || {
        yield ();
    };
}
