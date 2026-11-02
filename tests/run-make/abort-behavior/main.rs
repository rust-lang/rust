#![feature(abort_immediate)]

fn main() {
    let arg = std::env::args().skip(1).next().expect("no argument passed");
    match arg.as_str() {
        "abort" => std::process::abort(),
        "crash" => std::process::crash(),
        _ => panic!("unrecognized command {arg}"),
    }
}
