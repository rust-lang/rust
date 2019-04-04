//! ```cargo
//! [dependencies]
//! ar = "0.6.2"
//! ```

use std::io::Read;

// 64 gives Invalid file size field in entry header
// 32 gives unexpected EOF in the middle of archive entry header
const METADATA_LEN: usize = 64;

fn main() {
    let mut builder = ar::Builder::new(std::fs::File::create("test.a").expect("create"));

    // Remove this append and there is no problem.
    let header = ar::Header::new(b"core-fc675.rcgu.o".to_vec(), 0);
    // Remove any of the characters in the filename and ! from will show up in the error message.
    // Making it shorter than 17 chars will fix the problem though.

    builder.append(&header, &mut (&[] as &[u8])).expect("add rcgu");

    let mut buf: Vec<u8> = vec!['!' as u8; 28];
    buf.extend(b"hello worl");
    buf.extend(&['*' as u8; 26] as &[u8]);
    assert!(buf.len() >= METADATA_LEN);

    let header = ar::Header::new(b"rust.metadata.bin".to_vec(), METADATA_LEN as u64);
    builder.append(&header, &mut (&buf[0..METADATA_LEN])).expect("add meta");

    std::mem::drop(builder);

    // Remove this ranlib invocation and there is no problem.
    /*assert!(
        std::process::Command::new("ranlib")
            .arg("test.a")
            .status()
            .expect("Couldn't run ranlib")
            .success()
    );*/

    let mut archive = ar::Archive::new(std::fs::File::open("test.a").expect("open"));
    while let Some(entry) = archive.next_entry() {
        entry.unwrap();
    }
}
