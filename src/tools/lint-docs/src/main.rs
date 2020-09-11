use std::error::Error;
use std::path::PathBuf;

fn main() {
    if let Err(e) = doit() {
        println!("error: {}", e);
        std::process::exit(1);
    }
}

fn doit() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut src_path = None;
    let mut out_path = None;
    let mut rustc_path = None;
    let mut verbose = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--src" => {
                src_path = match args.next() {
                    Some(s) => Some(PathBuf::from(s)),
                    None => return Err("--src requires a value".into()),
                };
            }
            "--out" => {
                out_path = match args.next() {
                    Some(s) => Some(PathBuf::from(s)),
                    None => return Err("--out requires a value".into()),
                };
            }
            "--rustc" => {
                rustc_path = match args.next() {
                    Some(s) => Some(PathBuf::from(s)),
                    None => return Err("--rustc requires a value".into()),
                };
            }
            "-v" | "--verbose" => verbose = true,
            s => return Err(format!("unexpected argument `{}`", s).into()),
        }
    }
    if src_path.is_none() {
        return Err("--src must be specified to the directory with the compiler source".into());
    }
    if out_path.is_none() {
        return Err("--out must be specified to the directory with the lint listing docs".into());
    }
    if rustc_path.is_none() {
        return Err("--rustc must be specified to the path of rustc".into());
    }
    lint_docs::extract_lint_docs(
        &src_path.unwrap(),
        &out_path.unwrap(),
        &rustc_path.unwrap(),
        verbose,
    )
}
