#![feature(cfg_eval)]
#![feature(stmt_expr_attributes)]

fn main() {
    let _ = #[cfg_eval] #[cfg(false)] 0;
    //~^ ERROR `cfg` is not supported in this position
    //~| ERROR macro attributes on expressions are unstable
    let _ = #[cfg_eval] #[cfg(true)] 1;
    //~^ ERROR `cfg` is not supported in this position
    //~| ERROR macro attributes on expressions are unstable
}
