use std::path::PathBuf;

use object::pod::slice_from_all_bytes;
use object::read::pe::{ImageNtHeaders, PeFile, PeFile32, PeFile64};
use object::{FileKind, Object, pe};
use run_make_support::{bin_name, is_windows_msvc, object, rfs, rustc};

// Returns the TimeDateStamp values from every IMAGE_DEBUG_DIRECTORY entry
// in a parsed PE image.
// Shared by both PE32 and PE32+ binaries.
fn timestamps<Pe: ImageNtHeaders>(obj: &PeFile<'_, Pe>) -> Vec<u32> {
    let data_dir =
        obj.data_directory(pe::IMAGE_DIRECTORY_ENTRY_DEBUG).expect("no debug directory found");

    let debug_data =
        data_dir.data(obj.data(), &obj.section_table()).expect("failed to read debug directory");

    let debug_dirs = slice_from_all_bytes::<pe::ImageDebugDirectory>(debug_data)
        .expect("invalid IMAGE_DEBUG_DIRECTORY");

    let mut stamps = Vec::new();

    // COFF file header TimeDateStamp.
    stamps.push(obj.nt_headers().file_header().time_date_stamp.get(object::LittleEndian));

    // IMAGE_DEBUG_DIRECTORY TimeDateStamp values.
    stamps.extend(debug_dirs.iter().map(|d| d.time_date_stamp.get(object::LittleEndian)));

    stamps
}

fn main() {
    if !is_windows_msvc() {
        return;
    }

    // Compile the test crate and collect the
    // IMAGE_DEBUG_DIRECTORY timestamps from the resulting executable.
    let build = || -> Vec<u32> {
        rustc().input("main.rs").arg("-Clink-arg=/Brepro").output(bin_name("brepro-test")).run();

        let bytes = rfs::read(bin_name("brepro-test"));

        // Parse the generated executable according to its PE format so the
        // test works for both 32-bit and 64-bit MSVC targets.
        match FileKind::parse(bytes.as_slice()).unwrap() {
            FileKind::Pe32 => {
                let obj = PeFile32::parse(bytes.as_slice()).unwrap();
                timestamps(&obj)
            }
            FileKind::Pe64 => {
                let obj = PeFile64::parse(bytes.as_slice()).unwrap();
                timestamps(&obj)
            }
            kind => panic!("unexpected file kind: {kind:?}"),
        }
    };

    // First build.
    let stamps_a = build();

    // Remove the generated PDB so the second build starts fresh.
    rfs::remove_file(PathBuf::from(bin_name("brepro-test")).with_extension("pdb"));

    // Second build.
    let stamps_b = build();

    assert!(!stamps_a.is_empty(), "no IMAGE_DEBUG_DIRECTORY entries found");

    // /Brepro should make the linker emit deterministic timestamps across
    // identical builds.
    assert_eq!(stamps_a, stamps_b, "TimeDateStamp differs between identical builds");
}
