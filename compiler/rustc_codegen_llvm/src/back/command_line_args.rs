#[cfg(test)]
mod tests;

/// Joins command-line arguments into a single space-separated string, quoting
/// and escaping individual arguments as necessary.
///
/// The result is intended to be informational, for embedding in debug metadata,
/// and might not be properly quoted/escaped for actual command-line use.
pub(crate) fn quote_command_line_args(args: &[String]) -> String {
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
