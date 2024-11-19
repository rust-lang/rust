use rustc_feature::UnstableFeatures;

use crate::EarlyDiagCtxt;
use crate::config::{OptionStability, RustcOptGroup};

pub fn is_unstable_enabled(matches: &getopts::Matches) -> bool {
    match_is_nightly_build(matches)
        && matches.opt_strs("Z").iter().any(|x| *x == "unstable-options")
}

pub fn match_is_nightly_build(matches: &getopts::Matches) -> bool {
    is_nightly_build(matches.opt_str("crate-name").as_deref())
}

fn is_nightly_build(krate: Option<&str>) -> bool {
    UnstableFeatures::from_environment(krate).is_nightly_build()
}

pub fn check_nightly_options(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    flags: &[RustcOptGroup],
) {
    let has_z_unstable_option = matches.opt_strs("Z").iter().any(|x| *x == "unstable-options");
    let really_allows_unstable_options = match_is_nightly_build(matches);
    let mut nightly_options_on_stable = 0;

    for opt in flags.iter() {
        if opt.stability == OptionStability::Stable {
            continue;
        }
        if !matches.opt_present(opt.name) {
            continue;
        }
        if opt.name != "Z" && !has_z_unstable_option {
            early_dcx.early_fatal(format!(
                "the `-Z unstable-options` flag must also be passed to enable the flag `{}`",
                opt.name
            ));
        }
        if really_allows_unstable_options {
            continue;
        }
        match opt.stability {
            OptionStability::Unstable => {
                nightly_options_on_stable += 1;
                let msg =
                    format!("the option `{}` is only accepted on the nightly compiler", opt.name);
                let _ = early_dcx.early_err(msg);
            }
            OptionStability::Stable => {}
        }
    }
    if nightly_options_on_stable > 0 {
        early_dcx.early_help("consider switching to a nightly toolchain: `rustup default nightly`");
        early_dcx.early_note("selecting a toolchain with `+toolchain` arguments require a rustup proxy; see <https://rust-lang.github.io/rustup/concepts/index.html>");
        early_dcx.early_note("for more information about Rust's stability policy, see <https://doc.rust-lang.org/book/appendix-07-nightly-rust.html#unstable-features>");
        early_dcx.early_fatal(format!(
            "{} nightly option{} were parsed",
            nightly_options_on_stable,
            if nightly_options_on_stable > 1 { "s" } else { "" }
        ));
    }
}
