#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate languageserver_types;
extern crate drop_bomb;
extern crate crossbeam_channel;
extern crate libeditor;
extern crate libanalysis;

mod io;
mod caps;
mod req;
mod dispatch;

use languageserver_types::InitializeResult;
use libanalysis::WorldState;
use self::io::{Io, RawMsg};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

fn main() -> Result<()> {
    let mut io = Io::from_stdio();
    initialize(&mut io)?;
    io.stop()?;
    Ok(())
}

fn initialize(io: &mut Io) -> Result<()> {
    loop {
        match io.recv()? {
            RawMsg::Request(req) => {
                if let Some((_params, resp)) = dispatch::expect::<req::Initialize>(io, req)? {
                    resp.result(io, InitializeResult {
                        capabilities: caps::SERVER_CAPABILITIES
                    })?;
                    match io.recv()? {
                        RawMsg::Notification(n) => {
                            if n.method != "initialized" {
                                bail!("expected initialized notification");
                            }
                        }
                        _ => {
                            bail!("expected initialized notification");
                        }
                    }
                    return initialized(io);
                }
            }
            RawMsg::Notification(n) => {
                bail!("expected initialize request, got {:?}", n)
            }
            RawMsg::Response(res) => {
                bail!("expected initialize request, got {:?}", res)
            }
        }
    }
}

fn initialized(io: &mut Io) -> Result<()> {
    eprintln!("initialized");
    let world = WorldState::new();
    loop {
        match io.recv()? {
            RawMsg::Request(req) => {
                let world = world.snapshot();
                if let Some((params, resp)) = dispatch::expect::<req::SyntaxTree>(io, req)? {
                    resp.respond_with(io, || {
                        let path = params.text_document.uri.to_file_path()
                            .map_err(|()| format_err!("invalid path"))?;
                        let file = world.file_syntax(&path)?;
                        Ok(libeditor::syntax_tree(&file))
                    })?
                }
            }
            msg => {
                eprintln!("msg = {:?}", msg);
            }
        }
    }
}

