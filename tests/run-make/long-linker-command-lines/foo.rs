use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

fn write_test_case(file: &Path, n: usize) -> HashSet<String> {
    let mut libs = HashSet::new();
    let mut f = BufWriter::new(File::create(&file).unwrap());
    let mut prefix = String::new();
    for _ in 0..n {
        prefix.push_str("foo");
    }
    for i in 0..n {
        writeln!(f, "#[link(name = \"S{}{}S\")]", prefix, i).unwrap();
        libs.insert(format!("{}{}", prefix, i));
    }
    writeln!(f, "extern \"C\" {{}}\nfn main() {{}}").unwrap();
    f.into_inner().unwrap();

    libs
}

fn read_linker_args(path: &Path) -> String {
    let contents = fs::read(path).unwrap();
    if cfg!(target_env = "msvc") {
        let mut i = contents.chunks(2).map(|c| c[0] as u16 | ((c[1] as u16) << 8));
        assert_eq!(i.next(), Some(0xfeff), "Expected UTF-16 BOM");
        String::from_utf16(&i.collect::<Vec<u16>>()).unwrap()
    } else {
        String::from_utf8(contents).unwrap()
    }
}

fn main() {
    let ok = PathBuf::from("ok");
    if env::var("YOU_ARE_A_LINKER").is_ok() {
        if let Some(file) = env::args_os().find(|a| a.to_string_lossy().contains("@")) {
            let file = file.to_str().expect("non-utf8 file argument");
            fs::copy(&file[1..], &ok).unwrap();
        }
        return;
    }

    let rustc = env::var_os("RUSTC").unwrap();
    let me_as_linker = format!("linker={}", env::current_exe().unwrap().display());
    for i in (1..).map(|i| i * 100) {
        println!("attempt: {}", i);
        let file = PathBuf::from("bar.rs");
        let mut expected_libs = write_test_case(&file, i);

        drop(fs::remove_file(&ok));
        let output = Command::new(&rustc)
            .arg(&file)
            .arg("-C")
            .arg(&me_as_linker)
            .env("YOU_ARE_A_LINKER", "1")
            .output()
            .unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "status: {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                stderr
                    .lines()
                    .map(|l| {
                        if l.len() > 200 {
                            format!("{}...\n", &l[..200])
                        } else {
                            format!("{}\n", l)
                        }
                    })
                    .collect::<String>()
            );
        }

        if !ok.exists() {
            continue;
        }

        let linker_args = read_linker_args(&ok);
        for arg in linker_args.split('S') {
            expected_libs.remove(arg);
        }

        assert!(
            expected_libs.is_empty(),
            "expected but missing libraries: {:#?}\nlinker arguments: \n{}",
            expected_libs,
            linker_args,
        );

        break;
    }
}
