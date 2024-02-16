use std::env;
use std::error::Error;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut path: PathBuf =
        env::args_os().nth(1).expect("a path to the rust repository is required").into();
    path.push("library/std/src/sys/pal/windows/c");
    env::set_current_dir(&path)?;

    sort_bindings("bindings.txt")?;

    let info = windows_bindgen::bindgen(["--etc", "bindings.txt"])?;
    println!("{info}");

    let mut f = std::fs::File::options().append(true).open("windows_sys.rs")?;
    writeln!(&mut f, "// ignore-tidy-filelength")?;

    Ok(())
}

fn sort_bindings(file_name: &str) -> Result<(), Box<dyn Error>> {
    let mut f = fs::File::options().read(true).write(true).open(file_name)?;
    let mut bindings = String::new();
    f.read_to_string(&mut bindings)?;
    f.set_len(0)?;
    f.seek(SeekFrom::Start(0))?;

    let mut lines = bindings.split_inclusive('\n');
    for line in &mut lines {
        f.write(line.as_bytes())?;
        if line.contains("--filter") {
            break;
        }
    }
    let mut bindings = Vec::new();
    for line in &mut lines {
        if !line.trim().is_empty() {
            bindings.push(line);
        }
    }
    bindings.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    for line in bindings {
        f.write(line.as_bytes())?;
    }
    Ok(())
}
