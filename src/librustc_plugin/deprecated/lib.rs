#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(staged_api)]
#![unstable(feature = "rustc_plugin", issue = "29597")]
#![rustc_deprecated(since = "1.38.0", reason = "\
    import this through `rustc_driver::plugin` instead to make TLS work correctly. \
    See https://github.com/rust-lang/rust/issues/62717")]

pub use rustc_plugin_impl::*;
