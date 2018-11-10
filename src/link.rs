use std::fs::File;
use std::path::PathBuf;

use rustc::session::Session;

pub(crate) fn link_rlib(sess: &Session, res: &crate::CodegenResults, output_name: PathBuf) {
    let file = File::create(&output_name).unwrap();
    let mut builder = ar::Builder::new(file);

    // Add main object file
    let obj = res.artifact.emit().unwrap();
    builder
        .append(
            &ar::Header::new(b"data.o".to_vec(), obj.len() as u64),
            ::std::io::Cursor::new(obj),
        )
        .unwrap();

    // Non object files need to be added after object files, because ranlib will
    // try to read the native architecture from the first file, even if it isn't
    // an object file
    builder
        .append(
            &ar::Header::new(
                crate::metadata::METADATA_FILENAME.to_vec(),
                res.metadata.len() as u64,
            ),
            ::std::io::Cursor::new(res.metadata.clone()),
        )
        .unwrap();

    // Finalize archive
    std::mem::drop(builder);

    // Run ranlib to be able to link the archive
    let status = std::process::Command::new("ranlib")
        .arg(output_name)
        .status()
        .expect("Couldn't run ranlib");
    if !status.success() {
        sess.fatal(&format!("Ranlib exited with code {:?}", status.code()));
    }
}

pub(crate) fn link_bin(sess: &Session, res: &crate::CodegenResults, output_name: PathBuf) {
    // TODO: link executable
    let obj = res.artifact.emit().unwrap();
    std::fs::write(output_name, obj).unwrap();
}

/*
res.artifact
    .declare_with(
        &metadata_name,
        faerie::artifact::Decl::Data {
            global: true,
            writable: false,
        },
        res.metadata.clone(),
    )
    .unwrap();
*/
