//! Writing of the rustc metadata for dylibs

use rustc_middle::ty::TyCtxt;

use crate::backend::WriteMetadata;

// Adapted from https://github.com/rust-lang/rust/blob/da573206f87b5510de4b0ee1a9c044127e409bd3/src/librustc_codegen_llvm/base.rs#L47-L112
pub(crate) fn write_metadata<O: WriteMetadata>(tcx: TyCtxt<'_>, object: &mut O) {
    use snap::write::FrameEncoder;
    use std::io::Write;

    let metadata = tcx.encode_metadata();
    let mut compressed = rustc_metadata::METADATA_HEADER.to_vec();
    FrameEncoder::new(&mut compressed).write_all(&metadata.raw_data).unwrap();

    object.add_rustc_section(
        rustc_middle::middle::exported_symbols::metadata_symbol_name(tcx),
        compressed,
    );
}
