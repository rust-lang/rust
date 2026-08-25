use crate::io;
use crate::path::Path;
use crate::sys::pal::sgx_ineffective;

pub fn chdir(_: &Path) -> io::Result<()> {
    sgx_ineffective(())
}
