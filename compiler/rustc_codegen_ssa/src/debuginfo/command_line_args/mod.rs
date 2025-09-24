use std::sync::Arc;

use rustc_middle::middle::debuginfo::CommandLineArgsForDebuginfo;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;

#[cfg(test)]
mod tests;

pub(crate) fn provide(providers: &mut Providers) {
    providers.hooks.args_for_debuginfo = args_for_debuginfo;
}

/// Hook implementation for [`TyCtxt::args_for_debuginfo`].
fn args_for_debuginfo<'tcx>(tcx: TyCtxt<'tcx>) -> &'tcx Arc<CommandLineArgsForDebuginfo> {
    tcx.args_for_debuginfo_cache.get_or_init(|| {
        // Command-line information to be included in the target machine.
        // This seems to only be used for embedding in PDB debuginfo files.
        // FIXME(Zalathar): Maybe skip this for non-PDB targets?
        let argv0 = std::env::current_exe()
            .unwrap_or_default()
            .into_os_string()
            .into_string()
            .unwrap_or_default();
        let quoted_args = quote_command_line_args(&tcx.sess.expanded_args);

        // Self-profile counter for the number of bytes produced by command-line quoting.
        tcx.prof.artifact_size("quoted_command_line_args", "-", quoted_args.len() as u64);

        Arc::new(CommandLineArgsForDebuginfo { argv0, quoted_args })
    })
}

/// Joins command-line arguments into a single space-separated string, quoting
/// and escaping individual arguments as necessary.
///
/// The result is intended to be informational, for embedding in debug metadata,
/// and might not be properly quoted/escaped for actual command-line use.
fn quote_command_line_args(args: &[String]) -> String {
    // Start with a decent-sized buffer, since rustc invocations tend to be long.
    let mut buf = String::with_capacity(128);

    for arg in args {
        if !buf.is_empty() {
            buf.push(' ');
        }

        print_arg_quoted(&mut buf, arg);
    }

    buf
}

/// Equivalent to LLVM's `sys::printArg` with quoting always enabled
/// (see llvm/lib/Support/Program.cpp).
fn print_arg_quoted(buf: &mut String, arg: &str) {
    buf.reserve(arg.len() + 2);

    buf.push('"');
    for ch in arg.chars() {
        if matches!(ch, '"' | '\\' | '$') {
            buf.push('\\');
        }
        buf.push(ch);
    }
    buf.push('"');
}
