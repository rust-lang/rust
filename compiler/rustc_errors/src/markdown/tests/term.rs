use std::io::BufWriter;
use std::path::PathBuf;
use termcolor::{BufferWriter, ColorChoice};

use super::*;
use crate::markdown::MdStream;

const INPUT: &str = include_str!("input.md");
const OUTPUT_PATH: &[&str] = &[env!("CARGO_MANIFEST_DIR"), "src","markdown","tests","output.stdout"];

const TEST_WIDTH: usize = 80;

// We try to make some words long to create corner cases
const TXT: &str = r"Lorem ipsum dolor sit amet, consecteturadipiscingelit.
Fusce-id-urna-sollicitudin, pharetra nisl nec, lobortis tellus. In at
metus hendrerit, tincidunteratvel, ultrices turpis. Curabitur_risus_sapien,
porta-sed-nunc-sed, ultricesposuerelacus. Sed porttitor quis
dolor non venenatis. Aliquam ut. ";

const WRAPPED: &str = r"Lorem ipsum dolor sit amet, consecteturadipiscingelit. Fusce-id-urna-
sollicitudin, pharetra nisl nec, lobortis tellus. In at metus hendrerit,
tincidunteratvel, ultrices turpis. Curabitur_risus_sapien, porta-sed-nunc-sed,
ultricesposuerelacus. Sed porttitor quis dolor non venenatis. Aliquam ut. Lorem
    ipsum dolor sit amet, consecteturadipiscingelit. Fusce-id-urna-
    sollicitudin, pharetra nisl nec, lobortis tellus. In at metus hendrerit,
    tincidunteratvel, ultrices turpis. Curabitur_risus_sapien, porta-sed-nunc-
    sed, ultricesposuerelacus. Sed porttitor quis dolor non venenatis. Aliquam
    ut. Sample link lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet,
consecteturadipiscingelit. Fusce-id-urna-sollicitudin, pharetra nisl nec,
lobortis tellus. In at metus hendrerit, tincidunteratvel, ultrices turpis.
Curabitur_risus_sapien, porta-sed-nunc-sed, ultricesposuerelacus. Sed porttitor
quis dolor non venenatis. Aliquam ut. ";

#[test]
fn test_wrapping_write() {
    WIDTH.with(|w| w.set(TEST_WIDTH));
    let mut buf = BufWriter::new(Vec::new());
    let txt = TXT.replace("-\n","-").replace("_\n","_").replace('\n', " ").replace("    ", "");
    write_wrapping(&mut buf, &txt, 0, None).unwrap();
    write_wrapping(&mut buf, &txt, 4, None).unwrap();
    write_wrapping(
        &mut buf,
        "Sample link lorem ipsum dolor sit amet. ",
        4,
        Some("link-address-placeholder"),
    )
    .unwrap();
    write_wrapping(&mut buf, &txt, 0, None).unwrap();
    let out = String::from_utf8(buf.into_inner().unwrap()).unwrap();
    let out = out
        .replace("\x1b\\", "")
        .replace('\x1b', "")
        .replace("]8;;", "")
        .replace("link-address-placeholder", "");

    for line in out.lines() {
        assert!(line.len() <= TEST_WIDTH, "line length\n'{line}'")
    }

    assert_eq!(out, WRAPPED);
}

#[test]
fn test_output() {
    // Capture `--bless` when run via ./x
    let bless = std::env::var("RUSTC_BLESS").unwrap_or_default() == "1";
    let ast = MdStream::parse_str(INPUT);
    let bufwtr = BufferWriter::stderr(ColorChoice::Always);
    let mut buffer = bufwtr.buffer();
    ast.write_termcolor_buf(&mut buffer).unwrap();

    let mut blessed = PathBuf::new();
    blessed.extend(OUTPUT_PATH);

    if bless {
        std::fs::write(&blessed, buffer.into_inner()).unwrap();
        eprintln!("blessed output at {}", blessed.display());
    } else {
        let output = buffer.into_inner();
        if std::fs::read(blessed).unwrap() != output {
            // hack: I don't know any way to write bytes to the captured stdout
            // that cargo test uses
            let mut out = std::io::stdout();
            out.write_all(b"\n\nMarkdown output did not match. Expected:\n").unwrap();
            out.write_all(&output).unwrap();
            out.write_all(b"\n\n").unwrap();
            panic!("markdown output mismatch");
        }
    }
}
