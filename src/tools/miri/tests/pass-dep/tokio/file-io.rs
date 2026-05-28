//@compile-flags: -Zmiri-disable-isolation
//@only-target: linux # We only support tokio on Linux

use std::fs::remove_file;

use tokio::fs::{File, OpenOptions};
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};

#[path = "../../utils/mod.rs"]
mod utils;

#[tokio::main]
async fn main() {
    test_create_and_write().await.unwrap();
    test_create_and_read().await.unwrap();
}

async fn test_create_and_write() -> io::Result<()> {
    let path = utils::prepare("foo.txt");
    let mut file = File::create(&path).await?;

    // Write 10 bytes to the file.
    file.write_all(b"some bytes").await?;
    // For tokio's file I/O, `await` does not have its usual semantics of waiting until the
    // operation is completed, so we have to wait some more to make sure the write is completed.
    file.flush().await?;
    // Check that 10 bytes have been written.
    assert_eq!(file.metadata().await.unwrap().len(), 10);

    remove_file(&path).unwrap();
    Ok(())
}

async fn test_create_and_read() -> io::Result<()> {
    let bytes = b"more bytes";
    let path = utils::prepare_with_content("foo.txt", bytes);
    let mut file = OpenOptions::new().read(true).open(&path).await.unwrap();
    let mut buffer = vec![];

    // Read the whole file.
    file.read_to_end(&mut buffer).await?;
    assert_eq!(&buffer, b"more bytes");

    remove_file(&path).unwrap();
    Ok(())
}
