use crate::sys::pipe::{Pipes, anon_pipe};
use crate::{thread, time};

/// Test the synchronous fallback for overlapped I/O.
#[test]
fn overlapped_handle_fallback() {
    // Create some pipes. `ours` will be asynchronous.
    let Pipes { ours, theirs } = anon_pipe(true, false).unwrap();

    let async_readable = ours.into_handle();
    let sync_writeable = theirs.into_handle();

    thread::scope(|_| {
        thread::sleep(time::Duration::from_millis(100));
        sync_writeable.write(b"hello world!").unwrap();
    });

    // The pipe buffer starts empty so reading won't complete synchronously unless
    // our fallback path works.
    let mut buffer = [0u8; 1024];
    async_readable.read(&mut buffer).unwrap();
}
