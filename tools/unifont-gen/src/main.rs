use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::collections::BTreeMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let unifont_hex_path = project_root.join("assets/fonts/unifont.hex");
    let output_path = project_root.join("userspace/bloom/src/unifont_subset.hex");

    // Target codepoints from the original script
    let mut targets = Vec::new();
    for cp in 0x20..0x7F {
        targets.push(cp);
    }
    targets.extend_from_slice(&[
        0x2303, // ⌃
        0x2325, // ⌥
        0x21E7, // ⇧
        0x2318, // ⌘
        0x23CE, // ⏎
        0x232B, // ⌫
        0x21E5, // ⇥
        0x238B, // ⎋
        0x2190, // ←
        0x2191, // ↑
        0x2192, // →
        0x2193, // ↓
    ]);

    let mut glyph_data = BTreeMap::new();

    if !Path::new(unifont_hex_path).exists() {
        eprintln!("Error: unifont.hex not found at {}", unifont_hex_path);
        std::process::exit(1);
    }

    let file = File::open(unifont_hex_path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() != 2 {
            continue;
        }

        if let Ok(cp) = u32::from_str_radix(parts[0], 16) {
            if targets.contains(&cp) {
                glyph_data.insert(cp, line.clone());
            }
        }
    }

    let mut output = File::create(output_path)?;
    for (_, line) in glyph_data {
        writeln!(output, "{}", line)?;
    }

    println!("Generated subset with {} glyphs at {}", targets.len(), output_path);

    Ok(())
}
