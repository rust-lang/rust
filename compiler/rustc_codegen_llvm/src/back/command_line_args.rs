#[cfg(test)]
mod tests;

/// Joins command-line arguments into a single space-separated string, quoting
/// and escaping individual arguments as necessary.
///
/// The result is intended to be informational, for embedding in debug metadata,
/// and might not be properly quoted/escaped for actual command-line use.
pub(crate) fn quote_command_line_args(args: &[String]) -> String {
    // The characters we care about quoting are all ASCII, so we can get some free
    // performance by performing the quoting step on bytes instead of characters.
    //
    // Non-ASCII bytes will be copied as-is, so the result is still UTF-8.

    // Calculate an adequate buffer size, assuming no escaping will be needed.
    // The `+ 3` represents an extra space and pair of quotes per arg.
    let capacity_estimate = args.iter().map(|arg| arg.len() + 3).sum::<usize>();
    let mut buf = Vec::with_capacity(capacity_estimate);

    for arg in args {
        if !buf.is_empty() {
            buf.push(b' ');
        }

        print_arg_quoted(&mut buf, arg);
    }

    // Converting back to String isn't strictly necessary, since the bytes are
    // only passed to LLVM which doesn't care, but validating should be cheap
    // and it's nice to have some assurance that we didn't mess up.
    String::from_utf8(buf).expect("quoted args should still be UTF-8")
}

/// Equivalent to LLVM's `sys::printArg` with quoting always enabled
/// (see llvm/lib/Support/Program.cpp).
fn print_arg_quoted(buf: &mut Vec<u8>, arg: &str) {
    buf.reserve(arg.len() + 2);

    buf.push(b'"');
    for &byte in arg.as_bytes() {
        if matches!(byte, b'"' | b'\\' | b'$') {
            buf.push(b'\\');
        }
        buf.push(byte);
    }
    buf.push(b'"');
}
