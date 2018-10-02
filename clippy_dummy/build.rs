extern crate term;

fn main() {
    if let Err(_) = foo() {
        eprintln!("error: Clippy is no longer available via crates.io\n");
        eprintln!("help: please run `rustup component add clippy-preview` instead");
    }
    std::process::exit(1);
}

fn foo() -> Result<(), ()> {
    let mut t = term::stderr().ok_or(())?;

    t.attr(term::Attr::Bold).map_err(|_| ())?;
    t.fg(term::color::RED).map_err(|_| ())?;
    write!(t, "\nerror: ").map_err(|_| ())?;


    t.reset().map_err(|_| ())?;
    t.fg(term::color::WHITE).map_err(|_| ())?;
    writeln!(t, "Clippy is no longer available via crates.io\n").map_err(|_| ())?;


    t.attr(term::Attr::Bold).map_err(|_| ())?;
    t.fg(term::color::GREEN).map_err(|_| ())?;
    write!(t, "help: ").map_err(|_| ())?;


    t.reset().map_err(|_| ())?;
    t.fg(term::color::WHITE).map_err(|_| ())?;
    write!(t, "please run `").map_err(|_| ())?;

    t.attr(term::Attr::Bold).map_err(|_| ())?;
    write!(t, "rustup component add clippy-preview").map_err(|_| ())?;

    t.reset().map_err(|_| ())?;
    t.fg(term::color::WHITE).map_err(|_| ())?;
    writeln!(t, "` instead").map_err(|_| ())?;

    t.reset().map_err(|_| ())?;
    Ok(())
}