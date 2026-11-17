//! The function that the `probe=added-unused` and `probe=added` settings of
//! `-Z raise-optimize-speed` add to every crate.
//!
//! A crate that holds a function marked `#[optimize(speed)]` pays for the module that such
//! functions are compiled in. To measure the least that this can cost, on crates that mark no
//! function, the compiler adds an empty marked function to each crate.
//!
//! With `probe=added-unused`, nothing refers to the function. The collector takes it for a root
//! all the same and the partitioner keeps it visible to the crate's other modules, so code is
//! emitted for it; a program that links the crate has no reason to load its module. That is
//! what a crate pays for holding a marked function.
//!
//! With `probe=added`, a `#[used]` static refers to the function, and the partitioner places
//! the static with it. A program that links the crate then loads the function's module, on
//! the probe's account and no other, as it would if it reached a marked function that the
//! crate's own code calls. That is what a crate pays when the programs that link it use one.

use rustc_ast as ast;
use rustc_ast::attr;
use rustc_errors::Diag;
use rustc_parse::parser::{AllowConstBlockItems, ForceCollect};
use rustc_parse::{parse_in, source_str_to_stream};
use rustc_session::Session;
use rustc_session::config::{RAISE_OPTIMIZE_SPEED_PROBE_FILE, RaiseOptimizeSpeedProbe};
use rustc_span::{FileName, sym};

/// The function alone. It is in an unnamed constant so that it can neither be named by the
/// crate's own items nor collide with them, and its name, which is
/// `RAISE_OPTIMIZE_SPEED_UNUSED_PROBE_FN`, begins with an underscore so that it is not reported
/// as dead code. It carries no `#[optimize(speed)]`: an attribute written here would be resolved
/// from the crate root like any other, and a crate with an attribute macro or a derive named
/// `optimize` in scope there would not compile with the probe. `codegen_fn_attrs` marks the
/// function instead, which it knows by its name and its file.
const PROBE_UNUSED: &str = "\
const _: () = {
    fn _probe() {}
};";

/// The function and the static that refers to it. Neither carries an attribute, for the reason
/// above: `codegen_fn_attrs` marks the function as it marks the one above, and the static as
/// `#[used]` would mark it.
const PROBE_USED: &str = "\
const _: () = {
    fn probe() {}
    static PROBE: fn() = probe;
};";

/// Adds the probe's items to the crate. `pre_configured_attrs` is the crate's attributes with
/// `cfg_attr` expanded, as the injection of the standard library takes them.
pub fn inject(krate: &mut ast::Crate, pre_configured_attrs: &[ast::Attribute], sess: &Session) {
    let Some(raise) = sess.raise_optimize_speed() else { return };
    let source = match raise.probe {
        RaiseOptimizeSpeedProbe::AddedUnused => PROBE_UNUSED,
        RaiseOptimizeSpeedProbe::Added => PROBE_USED,
        RaiseOptimizeSpeedProbe::Off | RaiseOptimizeSpeedProbe::Chosen => return,
    };
    // No code is emitted for the crate, so the function would cost nothing to hold.
    if !sess.opts.output_types.should_codegen()
        // The static needs lang items that such a crate need not define, and the two
        // settings add their function to the same crates.
        || attr::contains_name(pre_configured_attrs, sym::no_core)
    {
        return;
    }
    let parse = || -> Result<Option<Box<ast::Item>>, Vec<Diag<'_>>> {
        let tokens = source_str_to_stream(
            &sess.psess,
            FileName::Custom(RAISE_OPTIMIZE_SPEED_PROBE_FILE.to_string()),
            source.to_string(),
            None,
        )?;
        parse_in(&sess.psess, tokens, "<optimize(speed) probe>", |p| {
            p.parse_item(ForceCollect::No, AllowConstBlockItems::No)
        })
        .map_err(|e| vec![e])
    };
    match parse() {
        Ok(item) => krate.items.extend(item),
        Err(errs) => {
            for err in errs {
                err.emit();
            }
        }
    }
}
