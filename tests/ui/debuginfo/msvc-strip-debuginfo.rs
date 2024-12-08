//@ compile-flags: -C strip=debuginfo
//@ only-msvc
//@ run-pass

use std::path::Path;

pub fn is_related_pdb<P: AsRef<Path>>(path: &P, exe: &P) -> bool {
    let (exe, path) = (exe.as_ref(), path.as_ref());

    path.extension()
        .map(|x| x.to_ascii_lowercase())
        .is_some_and(|x| x == "pdb")
        && path.file_stem() == exe.file_stem()
}

pub fn main() {
    let curr_exe = std::env::current_exe().unwrap();
    let curr_dir = curr_exe.parent().unwrap();

    let entries = std::fs::read_dir(curr_dir).unwrap();

    assert!(entries
        .map_while(|x| x.ok())
        .find(|x| is_related_pdb(&x.path(), &curr_exe))
        .is_some());
}
