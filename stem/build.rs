use std::{env, fs, path::PathBuf};

use pciids::{build_tables, filter_vendors, parse_pci_ids, render_rust, Mode};

fn main() {
    println!("cargo:rerun-if-env-changed=PCI_IDS_MODE");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let pci_ids_path = manifest_dir.join("..").join("assets/pci/pci.ids");
    println!("cargo:rerun-if-changed={}", pci_ids_path.display());

    let contents = fs::read_to_string(&pci_ids_path).expect("failed to read assets/pci/pci.ids");

    let parsed = parse_pci_ids(&contents);
    let mode = Mode::from_env(env::var("PCI_IDS_MODE").ok());
    let filtered = filter_vendors(&parsed.vendors, mode);
    let tables = build_tables(&filtered);
    let rust = render_rust(&tables, &parsed.snapshot, mode);

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("pci_ids_gen.rs");
    fs::write(&out_path, rust).expect("failed to write generated PCI ID table");
}
