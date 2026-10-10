//@ check-pass

#![allow(unexpected_cfgs)] // since we different cfgs

macro_rules! mac {
    {} => {
        #[cfg(attr)]
        mod m {
            #[lang_item]
            fn f() {}

            #[cfg_attr(target_thread_local, custom)]
            fn g() {}
        }

        #[cfg(attr)]
        unconfigured_invocation!();
    }
}

mac! {}

fn main() {}
