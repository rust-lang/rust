//! Large archive example.
//!
//! This creates several C files with a bunch of global arrays. The goal is to
//! create an rlib that is over 4GB in size so that the LLVM archiver creates
//! a /SYM64/ entry instead of /.
//!
//! It compiles the C files to .o files, and then uses `ar` to collect them
//! into a static library. It creates `foo.rs` with references to all the C
//! arrays, and then uses `rustc` to build an rlib with that static
//! information. It then creates `bar.rs` which links the giant libfoo.rlib,
//! which should fail since it can't read the large libfoo.rlib file.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;

// Number of object files to create.
const NOBJ: u32 = 12;
// Make the filename longer than 16 characters to force names to be placed in //
const PREFIX: &str = "abcdefghijklmnopqrstuvwxyz";

fn main() {
    let tmpdir = std::path::PathBuf::from(env::args_os().nth(1).unwrap());
    let mut foo_rs = File::create(tmpdir.join("foo.rs")).unwrap();
    write!(foo_rs, "extern \"C\" {{\n").unwrap();
    for obj in 0..NOBJ {
        let filename = tmpdir.join(&format!("{}{}.c", PREFIX, obj));
        let f = File::create(&filename).unwrap();
        let mut buf = BufWriter::new(f);
        write!(buf, "#include<stdint.h>\n").unwrap();
        for n in 0..50 {
            write!(buf, "int64_t FOO_{}_{}[] = {{\n", obj, n).unwrap();
            for x in 0..1024 {
                for y in 0..1024 {
                    write!(buf, "{},", (obj + n + x + y) % 10).unwrap();
                }
                write!(buf, "\n").unwrap();
            }
            write!(buf, "}};\n").unwrap();
            write!(foo_rs, "    pub static FOO_{}_{}: [i64; 1024*1024];\n", obj, n).unwrap();
        }
        drop(buf);
        println!("compile {:?}", filename);
        let status =
            Command::new("cc").current_dir(&tmpdir).arg("-c").arg(&filename).status().unwrap();
        if !status.success() {
            panic!("failed: {:?}", status);
        }
    }
    write!(foo_rs, "}}\n").unwrap();
    drop(foo_rs);
    let mut cmd = Command::new("ar");
    cmd.arg("-crs");
    cmd.arg(tmpdir.join("libfoo.a"));
    for obj in 0..NOBJ {
        cmd.arg(tmpdir.join(&format!("{}{}.o", PREFIX, obj)));
    }
    println!("archiving: {:?}", cmd);
    let status = cmd.status().unwrap();
    if !status.success() {
        panic!("failed: {:?}", status);
    }
}
