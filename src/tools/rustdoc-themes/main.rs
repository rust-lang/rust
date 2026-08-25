use std::env::args;
use std::fs::{File, create_dir_all};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, exit};

fn get_themes<P: AsRef<Path>>(style_path: P) -> Vec<String> {
    let mut ret = Vec::with_capacity(10);

    const BEGIN_THEME_MARKER: &'static str = "/* Begin theme: ";
    const END_THEME_MARKER: &'static str = "/* End theme: ";

    let timestamp =
        std::time::SystemTime::UNIX_EPOCH.elapsed().expect("time is after UNIX epoch").as_millis();

    let mut in_theme = None;
    create_dir_all("build/tmp").expect("failed to create temporary test directory");
    for line in BufReader::new(File::open(style_path).expect("read rustdoc.css failed")).lines() {
        let line = line.expect("read line from rustdoc.css failed");
        let line = line.trim();
        if line.starts_with(BEGIN_THEME_MARKER) {
            let theme_name = &line[BEGIN_THEME_MARKER.len()..].trim().trim_end_matches("*/").trim();
            let filename = format!("build/tmp/rustdoc.bootstrap.{timestamp}.{theme_name}.css");
            in_theme = Some(BufWriter::new(
                File::create(&filename).expect("failed to create temporary test css file"),
            ));
            ret.push(filename);
        }
        if let Some(in_theme) = in_theme.as_mut() {
            in_theme.write_all(line.as_bytes()).expect("write to temporary test css file");
            in_theme.write_all(b"\n").expect("write to temporary test css file");
        }
        if line.starts_with(END_THEME_MARKER) {
            in_theme = None;
        }
    }
    ret
}

fn main() {
    let argv: Vec<String> = args().collect();

    if argv.len() < 3 {
        eprintln!("Needs rustdoc binary path");
        exit(1);
    }
    let rustdoc_bin = &argv[1];
    let style_path = &argv[2];
    let themes = get_themes(&style_path);
    if themes.is_empty() {
        eprintln!("No themes found in \"{}\"...", style_path);
        exit(1);
    }
    let arg_name = "--check-theme".to_owned();
    let status = Command::new(rustdoc_bin)
        .args(&themes.iter().flat_map(|t| vec![&arg_name, t].into_iter()).collect::<Vec<_>>())
        .status()
        .expect("failed to execute child");
    if !status.success() {
        exit(1);
    }
}
