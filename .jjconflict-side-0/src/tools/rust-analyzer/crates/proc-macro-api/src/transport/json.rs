//! Protocol functions for json.
use std::io::{self, BufRead, Write};

use serde::{Serialize, de::DeserializeOwned};

pub(crate) fn read<'a, R: BufRead + ?Sized>(
    inp: &mut R,
    buf: &'a mut String,
) -> io::Result<Option<&'a mut String>> {
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

pub(crate) fn write<W: Write + ?Sized>(out: &mut W, buf: &String) -> io::Result<()> {
    tracing::debug!("> {}", buf);
    out.write_all(buf.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()
}

pub(crate) fn encode<T: Serialize>(msg: &T) -> io::Result<String> {
    Ok(serde_json::to_string(msg)?)
}

pub(crate) fn decode<T: DeserializeOwned>(buf: &mut str) -> io::Result<T> {
    let mut deserializer = serde_json::Deserializer::from_str(buf);
    // Note that some proc-macro generate very deep syntax tree
    // We have to disable the current limit of serde here
    deserializer.disable_recursion_limit();
    Ok(T::deserialize(&mut deserializer)?)
}
