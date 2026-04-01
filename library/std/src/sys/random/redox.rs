use crate::fs::File;
use crate::io::Read;
use crate::sync::OnceLock;

static SCHEME: OnceLock<File> = OnceLock::new();

pub fn fill_bytes(bytes: &mut [u8]) {
    SCHEME
        .get_or_try_init(|| File::open("/scheme/rand"))
        .and_then(|mut scheme| scheme.read_exact(bytes))
        .expect("failed to generate random data");
}
