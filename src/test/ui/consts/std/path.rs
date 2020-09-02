// run-pass

use std::path::Prefix;

fn main() {
    const PREFIX : Prefix = Prefix::Disk(2);

    const IS_VERBATIM : bool = PREFIX.is_verbatim();
    assert!(!IS_VERBATIM);
}
