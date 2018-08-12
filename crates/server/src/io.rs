use std::{
    thread,
    io::{
        stdout, stdin,
        BufRead, Write,
    },
};
use serde_json::{Value, from_str, to_string};
use crossbeam_channel::{Receiver, Sender, bounded};

use Result;


#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RawMsg {
    Request(RawRequest),
    Notification(RawNotification),
    Response(RawResponse),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawRequest {
    pub id: u64,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawNotification {
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RawResponse {
    // JSON RPC allows this to be null if it was impossible
    // to decode the request's id. Ignore this special case
    // and just die horribly.
    pub id: u64,
    #[serde(default)]
    pub result: Value,
    #[serde(default)]
    pub error: Value,
}

struct MsgReceiver {
    chan: Receiver<RawMsg>,
    thread: Option<thread::JoinHandle<Result<()>>>,
}

impl MsgReceiver {
    fn recv(&mut self) -> Result<RawMsg> {
        match self.chan.recv() {
            Some(msg) => Ok(msg),
            None => {
                self.cleanup()?;
                unreachable!()
            }
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        self.thread
            .take()
            .ok_or_else(|| format_err!("MsgReceiver thread panicked"))?
            .join()
            .map_err(|_| format_err!("MsgReceiver thread panicked"))??;
        bail!("client disconnected")
    }

    fn stop(self) -> Result<()> {
        // Can't really self.thread.join() here, b/c it might be
        // blocking on read
        Ok(())
    }
}

struct MsgSender {
    chan: Sender<RawMsg>,
    thread: thread::JoinHandle<Result<()>>,
}

impl MsgSender {
    fn send(&mut self, msg: RawMsg) {
        self.chan.send(msg)
    }

    fn stop(self) -> Result<()> {
        drop(self.chan);
        self.thread.join()
            .map_err(|_| format_err!("MsgSender thread panicked"))??;
        Ok(())
    }
}

pub struct Io {
    receiver: MsgReceiver,
    sender: MsgSender,
}

impl Io {
    pub fn from_stdio() -> Io {
        let sender = {
            let (tx, rx) = bounded(16);
            MsgSender {
                chan: tx,
                thread: thread::spawn(move || {
                    let stdout = stdout();
                    let mut stdout = stdout.lock();
                    for msg in rx {
                        #[derive(Serialize)]
                        struct JsonRpc {
                            jsonrpc: &'static str,
                            #[serde(flatten)]
                            msg: RawMsg,
                        }
                        let text = to_string(&JsonRpc {
                            jsonrpc: "2.0",
                            msg,
                        })?;
                        write_msg_text(&mut stdout, &text)?;
                    }
                    Ok(())
                }),
            }
        };
        let receiver = {
            let (tx, rx) = bounded(16);
            MsgReceiver {
                chan: rx,
                thread: Some(thread::spawn(move || {
                    let stdin = stdin();
                    let mut stdin = stdin.lock();
                    while let Some(text) = read_msg_text(&mut stdin)? {
                        let msg: RawMsg = from_str(&text)?;
                        tx.send(msg);
                    }
                    Ok(())
                })),
            }
        };
        Io { receiver, sender }
    }

    pub fn send(&mut self, msg: RawMsg) {
        self.sender.send(msg)
    }

    pub fn recv(&mut self) -> Result<RawMsg> {
        self.receiver.recv()
    }

    pub fn receiver(&mut self) -> &mut Receiver<RawMsg> {
        &mut self.receiver.chan
    }

    pub fn cleanup_receiver(&mut self) -> Result<()> {
        self.receiver.cleanup()
    }

    pub fn stop(self) -> Result<()> {
        self.receiver.stop()?;
        self.sender.stop()?;
        Ok(())
    }
}


fn read_msg_text(inp: &mut impl BufRead) -> Result<Option<String>> {
    let mut size = None;
    let mut buf = String::new();
    loop {
        buf.clear();
        if inp.read_line(&mut buf)? == 0 {
            return Ok(None);
        }
        if !buf.ends_with("\r\n") {
            bail!("malformed header: {:?}", buf);
        }
        let buf = &buf[..buf.len() - 2];
        if buf.is_empty() {
            break;
        }
        let mut parts = buf.splitn(2, ": ");
        let header_name = parts.next().unwrap();
        let header_value = parts.next().ok_or_else(|| format_err!("malformed header: {:?}", buf))?;
        if header_name == "Content-Length" {
            size = Some(header_value.parse::<usize>()?);
        }
    }
    let size = size.ok_or_else(|| format_err!("no Content-Length"))?;
    let mut buf = buf.into_bytes();
    buf.resize(size, 0);
    inp.read_exact(&mut buf)?;
    let buf = String::from_utf8(buf)?;
    debug!("< {}", buf);
    Ok(Some(buf))
}

fn write_msg_text(out: &mut impl Write, msg: &str) -> Result<()> {
    debug!("> {}", msg);
    write!(out, "Content-Length: {}\r\n\r\n", msg.len())?;
    out.write_all(msg.as_bytes())?;
    out.flush()?;
    Ok(())
}
