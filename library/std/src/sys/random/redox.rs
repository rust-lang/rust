use crate::fs::File;
use crate::io::{BorrowedCursor, Read};
use crate::sync::OnceLock;

static SCHEME: OnceLock<File> = OnceLock::new();

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    SCHEME
        .get_or_try_init(|| File::open("/scheme/rand"))
        .and_then(|mut scheme| scheme.read_buf_exact(cursor))
        .expect("failed to generate random data");
}
