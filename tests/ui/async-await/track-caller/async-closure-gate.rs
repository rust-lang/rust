//@ edition:2021
//@ revisions: cls nofeat
//@[cls] check-pass

#![feature(stmt_expr_attributes)]
#![cfg_attr(cls, feature(closure_track_caller))]

fn main() {
    let _ = #[track_caller]
    //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async || {};
}

#[track_caller]
async fn foo() {
    let _ = #[track_caller]
    //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    async || {};
}

async fn foo2() {
    let _ = #[track_caller]
    //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    || {};
}

fn foo3() {
    let _ = async {
        let _ = #[track_caller]
        //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        || {};
    };
}

async fn foo4() {
    let _ = || {
        #[track_caller]
        //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        || {};
    };
}

fn foo5() {
    let _ = async {
        let _ = || {
            #[track_caller]
            //[nofeat]~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
            || {};
        };
    };
}
