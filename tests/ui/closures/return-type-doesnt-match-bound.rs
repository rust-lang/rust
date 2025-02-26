use std::error::Error;
use std::process::exit;

fn foo<F>(f: F) -> ()
where
    F: FnOnce() -> Result<(), Box<dyn Error>>,
{
    f().or_else(|e| -> ! { //~ ERROR to return
        eprintln!("{:?}", e);
        exit(1)
    });
}

fn bar<F>(f: F) -> ()
where
    F: FnOnce() -> Result<(), Box<dyn Error>>,
{
    let c = |e| -> ! { //~ ERROR to return
        eprintln!("{:?}", e);
        exit(1)
    };
    f().or_else(c);
}

fn main() {}
