#![feature(abort_immediate)]

fn main() {
    let arg = std::env::args().skip(1).next().expect("no argument passed");
    match arg.as_str() {
        "abort" => std::process::abort(),
        "abort_immediate" => std::process::abort_immediate(),
        _ => panic!("unrecognized command {}", arg),
    }
}
