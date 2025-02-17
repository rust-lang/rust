use std::error::Error;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::{env, fs};

/// 32-bit ARM is not supported by Microsoft so ARM types are not generated.
/// Therefore we need to inject a few types to make the bindings work.
const ARM32_SHIM: &str = r#"
#[cfg(target_arch = "arm")]
#[repr(C)]
pub struct WSADATA {
    pub wVersion: u16,
    pub wHighVersion: u16,
    pub szDescription: [u8; 257],
    pub szSystemStatus: [u8; 129],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: PSTR,
}
#[cfg(target_arch = "arm")]
pub enum CONTEXT {}
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let mut path: PathBuf =
        env::args_os().nth(1).expect("a path to the rust repository is required").into();
    path.push("library/std/src/sys/pal/windows/c");
    env::set_current_dir(&path)?;

    sort_bindings("bindings.txt")?;

    windows_bindgen::bindgen(["--etc", "bindings.txt"]);

    let mut f = std::fs::File::options().append(true).open("windows_sys.rs")?;
    f.write_all(ARM32_SHIM.as_bytes())?;
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
