// This is a test which attempts to blow out the system limit with how many
// arguments can be passed to a process. This'll successively call rustc with
// larger and larger argument lists in an attempt to find one that's way too
// big for the system at hand. This file itself is then used as a "linker" to
// detect when the process creation succeeds.
//
// Eventually we should see an argument that looks like `@` as we switch from
// passing literal arguments to passing everything in the file.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let tmpdir = PathBuf::from(env::var_os("TMPDIR").unwrap());
    let ok = tmpdir.join("ok");
    if env::var("YOU_ARE_A_LINKER").is_ok() {
        if let Some(file) = env::args_os().find(|a| a.to_string_lossy().contains("@")) {
            let file = file.to_str().expect("non-utf8 file argument");
            fs::copy(&file[1..], &ok).unwrap();
        }
        return
    }

    let rustc = env::var_os("RUSTC").unwrap_or("rustc".into());
    let me_as_linker = format!("linker={}", env::current_exe().unwrap().display());
    for i in (1..).map(|i| i * 100) {
        println!("attempt: {}", i);
        let file = tmpdir.join("bar.rs");
        let mut f = BufWriter::new(File::create(&file).unwrap());
        let mut lib_name = String::new();
        for _ in 0..i {
            lib_name.push_str("foo");
        }
        for j in 0..i {
            writeln!(f, "#[link(name = \"{}{}\")]", lib_name, j).unwrap();
        }
        writeln!(f, "extern {{}}\nfn main() {{}}").unwrap();
        f.into_inner().unwrap();

        drop(fs::remove_file(&ok));
        let output = Command::new(&rustc)
            .arg(&file)
            .arg("-C").arg(&me_as_linker)
            .arg("--out-dir").arg(&tmpdir)
            .env("YOU_ARE_A_LINKER", "1")
            .output()
            .unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("status: {}\nstdout:\n{}\nstderr:\n{}",
                   output.status,
                   String::from_utf8_lossy(&output.stdout),
                   stderr.lines().map(|l| {
                       if l.len() > 200 {
                           format!("{}...\n", &l[..200])
                       } else {
                           format!("{}\n", l)
                       }
                   }).collect::<String>());
        }

        if !ok.exists() {
            continue
        }

        let mut contents = Vec::new();
        File::open(&ok).unwrap().read_to_end(&mut contents).unwrap();

        for j in 0..i {
            let exp = format!("{}{}", lib_name, j);
            let exp = if cfg!(target_env = "msvc") {
                let mut out = Vec::with_capacity(exp.len() * 2);
                for c in exp.encode_utf16() {
                    // encode in little endian
                    out.push(c as u8);
                    out.push((c >> 8) as u8);
                }
                out
            } else {
                exp.into_bytes()
            };
            assert!(contents.windows(exp.len()).any(|w| w == &exp[..]));
        }

        break
    }
}
