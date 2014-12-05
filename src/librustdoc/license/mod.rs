use std::io::{Writer, IoResult, IoError};
use std::io::IoErrorKind::InvalidInput;
use flate;

mod text;

pub fn print(out: &mut Writer) -> IoResult<()> {
    match flate::inflate_bytes_zlib(text::COPYRIGHT) {
        Some(dat) => out.write(dat.as_slice()),
        None => Err(IoError { kind: InvalidInput, desc: "decompression error the COPYRIGHT file",
                              detail: None })
    }
}
