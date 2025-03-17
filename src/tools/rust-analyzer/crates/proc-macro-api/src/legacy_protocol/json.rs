//! Protocol functions for json.
use std::io::{self, BufRead, Write};

/// Reads a JSON message from the input stream.
pub fn read_json<'a>(
    inp: &mut impl BufRead,
    buf: &'a mut String,
) -> io::Result<Option<&'a String>> {
    loop {
        buf.clear();

        inp.read_line(buf)?;
        buf.pop(); // Remove trailing '\n'

        if buf.is_empty() {
            return Ok(None);
        }

        // Some ill behaved macro try to use stdout for debugging
        // We ignore it here
        if !buf.starts_with('{') {
            tracing::error!("proc-macro tried to print : {}", buf);
            continue;
        }

        return Ok(Some(buf));
    }
}

/// Writes a JSON message to the output stream.
pub fn write_json(out: &mut impl Write, msg: &str) -> io::Result<()> {
    tracing::debug!("> {}", msg);
    out.write_all(msg.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}
