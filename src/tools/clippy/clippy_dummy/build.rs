use term::color::{GREEN, RED, WHITE};
use term::{Attr, Error, Result};

fn main() {
    if foo().is_err() {
        eprintln!(
            "error: Clippy is no longer available via crates.io\n\n\
             help: please run `rustup component add clippy` instead"
        );
    }
    std::process::exit(1);
}

fn foo() -> Result<()> {
    let mut t = term::stderr().ok_or(Error::NotSupported)?;

    t.attr(Attr::Bold)?;
    t.fg(RED)?;
    write!(t, "\nerror: ")?;

    t.reset()?;
    t.fg(WHITE)?;
    writeln!(t, "Clippy is no longer available via crates.io\n")?;

    t.attr(Attr::Bold)?;
    t.fg(GREEN)?;
    write!(t, "help: ")?;

    t.reset()?;
    t.fg(WHITE)?;
    write!(t, "please run `")?;

    t.attr(Attr::Bold)?;
    write!(t, "rustup component add clippy")?;

    t.reset()?;
    t.fg(WHITE)?;
    writeln!(t, "` instead")?;

    t.reset()?;
    Ok(())
}
