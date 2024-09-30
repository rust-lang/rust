extern dyn crate libr;

use libr::*;

fn main() {
    assert_eq!(S::<S2>::foo(), 2);
}
