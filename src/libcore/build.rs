// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code generation for bytewise slice comparison. Target is to provide a compile time optimization
//! within a single macro which looks like:
//!
//! ```
//! macro_rules! slice_compare (
//!     ($a:expr, $b:expr, $len:expr) => {{
//!         match $len {
//!             1 => cmp!($a, $b, u8, 0),
//!             2 => cmp!($a, $b, u16, 0),
//!             3 => cmp!($a, $b, u16, 0) && cmp!($a, $b, u8, 2),
//!             4 => cmp!($a, $b, u32, 0),
//!             ...
//!         }
//!     }}
//! );
//! ```
//!
//! The supported slice length can be set by changing the `OPT_LEN` variable.

static OPT_LEN: usize = 256;

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

    for i in 1..OPT_LEN + 1 {
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
