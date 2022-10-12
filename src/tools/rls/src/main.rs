//! RLS stub.
//!
//! This is a small stub that replaces RLS to alert the user that RLS is no
//! longer available.

use serde::Deserialize;
use std::error::Error;
use std::io::BufRead;
use std::io::Write;

const ALERT_MSG: &str = "\
RLS is no longer available as of Rust 1.65.
Consider migrating to rust-analyzer instead.
See https://rust-analyzer.github.io/ for installation instructions.
";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[derive(Deserialize)]
struct Message {
    method: Option<String>,
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    let init = read_message(&mut stdin)?;
    if init.method.as_deref() != Some("initialize") {
        return Err(format!("expected initialize, got {:?}", init.method).into());
    }
    // No response, the LSP specification says that `showMessageRequest` may
    // be posted before during this phase.

    // message_type 1 is "Error"
    let alert = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "window/showMessageRequest",
        "params": {
            "message_type": "1",
            "message": ALERT_MSG
        }
    });
    write_message_raw(&mut stdout, serde_json::to_string(&alert).unwrap())?;

    loop {
        let message = read_message(&mut stdin)?;
        if message.method.as_deref() == Some("shutdown") {
            std::process::exit(0);
        }
    }
}

fn read_message_raw<R: BufRead>(reader: &mut R) -> Result<String, Box<dyn Error>> {
    let mut content_length: usize = 0;

    // Read headers.
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.is_empty() {
            return Err("remote disconnected".into());
        }
        if line == "\r\n" {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            let value = &line[15..].trim();
            content_length = usize::from_str_radix(value, 10)?;
        }
    }
    if content_length == 0 {
        return Err("no content-length".into());
    }

    let mut buffer = vec![0; content_length];
    reader.read_exact(&mut buffer)?;
    let content = String::from_utf8(buffer)?;

    Ok(content)
}

fn read_message<R: BufRead>(reader: &mut R) -> Result<Message, Box<dyn Error>> {
    let m = read_message_raw(reader)?;
    match serde_json::from_str(&m) {
        Ok(m) => Ok(m),
        Err(e) => Err(format!("failed to parse message {m}\n{e}").into()),
    }
}

fn write_message_raw<W: Write>(mut writer: W, output: String) -> Result<(), Box<dyn Error>> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", output.len(), output)?;
    writer.flush()?;
    Ok(())
}
