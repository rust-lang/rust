//@ edition:2021
//@ revisions: afn nofeat

#![feature(stmt_expr_attributes)]
#![cfg_attr(afn, feature(async_fn_track_caller))]

fn main() {
    let _ = #[track_caller] async || {
        //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    };
}

#[track_caller]
async fn foo() {
    let _ = #[track_caller] async || {
        //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    };
}

async fn foo2() {
    let _ = #[track_caller] || {
        //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
    };
}

fn foo3() {
    async {
        //~^ ERROR mismatched types
        let _ = #[track_caller] || {
            //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        };
    }
}

async fn foo4() {
    let _ = || {
        #[track_caller] || {
            //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
        };
    };
}

fn foo5() {
    async {
        //~^ ERROR mismatched types
        let _ = || {
            #[track_caller] || {
                //~^ ERROR `#[track_caller]` on closures is currently unstable [E0658]
            };
        };
    }
}
