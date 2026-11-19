#![feature(abort_immediate)]

fn main() {
    let arg = std::env::args().skip(1).next().expect("no argument passed");
    match arg.as_str() {
        "abort" => std::process::abort(),
        "crash" => std::process::crash(),
        "panic" => panic!("panic!"),
        "exit-0" => std::process::exit(0),
        "exit-1" => std::process::exit(1),
        _ => panic!("unrecognized command {arg}"),
    }
}
