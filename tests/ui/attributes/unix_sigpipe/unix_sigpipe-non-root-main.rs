//@ revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(unix_sigpipe))]

mod m {
    #[unix_sigpipe = "sig_dfl"] //~ error: `unix_sigpipe` attribute can only be used on root `fn main()`
    fn main() {}
}

fn main() {}
