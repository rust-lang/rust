use std::{env, iter, io};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::error::Error;

// Generates the code for the compile time optimization for bytewise slice comparison
pub fn main() {
    run().expect("Could not generate slice comparison source code.");
}

fn fill(indent: usize) -> String {
    iter::repeat(' ').take(indent).collect()
}

fn run() -> Result<(), Box<Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let dest_path = Path::new(&out_dir).join("memcmp_optimization.rs");
    let mut f = File::create(&dest_path)?;

    // Generate the slice comparison source code
    writeln!(f, "macro_rules! slice_compare (")?;
    writeln!(f, "{}($a:expr, $b:expr, $len:expr) => {{{{", fill(4))?;
    writeln!(f, "{}match $len {{", fill(8))?;

    for i in 1..257 {
        let mut bits = i * 8 as usize;
        let mut sizes = vec![8, 16, 32, 64];
        let mut offset = 0;

        write!(f, "{}{} => ", fill(12), i)?;
        while !sizes.is_empty() {
            let size = sizes.last().ok_or(io::Error::from(io::ErrorKind::Other))?.clone();
            if bits >= size {
                if offset > 0 {
                    write!(f, " && ")?;
                }
                write!(f, "cmp!($a, $b, u{}, {})", size, offset)?;
                bits = bits.checked_sub(size).ok_or(io::Error::from(io::ErrorKind::Other))?;
                offset += size / 8;
            } else {
                sizes.pop();
            }
            if bits == 0 {
                break;
            }
        }
        writeln!(f, ",")?;
    }

    writeln!(f,
             "{}_ => unsafe {{ memcmp($a, $b, $len) == 0 }},",
             fill(12))?;

    writeln!(f, "{}}}", fill(8))?;
    writeln!(f, "{}}}}}", fill(4))?;
    writeln!(f, ");")?;
    Ok(())
}
