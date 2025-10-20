//use std::collections::HashSet;
use std::path::PathBuf;

use build_helper::git::output_result;

//use tempfile::{NamedTempFile, tempdir_in};
use crate::diagnostics::TidyFlags;
use crate::walk::walk;

fn get_test_dir() -> PathBuf {
    let root_path = PathBuf::from(
        t!(output_result(std::process::Command::new("git").args(["rev-parse", "--show-toplevel"])))
            .trim(),
    );
    let test_dir = root_path.join("src/tools/tidy/src/tidy_flags/");
    test_dir
}

//fn get_untracked(root_path: &PathBuf) -> HashSet<PathBuf> {
//let untracked_files = match get_git_untracked_files(Some(&root_path)) {
//Ok(Some(untracked_paths)) => {
//untracked_paths.into_iter().map(|s| root_path.join(s)).collect()
//}
//_ => HashSet::new(),
//};
//untracked_files
//}

//#[test]
////Creates a temp untracked file and checks that there are more untracked files than
////previously.
//fn test_get_untracked() {
//let temp_dir = tempdir_in(get_test_dir()).unwrap();
//let temp_dir_path = temp_dir.path().to_path_buf();

//let num_untracked = get_untracked(&temp_dir_path).len();

//assert!(num_untracked == 0);

//let _new_file = tempfile::NamedTempFile::new_in(&temp_dir_path).unwrap();

//let new_num_untracked = get_untracked(&temp_dir_path).len();

//assert!(new_num_untracked == 1);
//}

//#[test]
////Various checks for the walk function and interactions with `TidyFlags`.
//fn test_tidy_walk() {
//let temp_dir = tempdir_in(get_test_dir()).unwrap();
//let temp_dir_path = temp_dir.path().to_path_buf();

//let _file_guards: Vec<NamedTempFile> = (0..5)
//.map(|_| tempfile::Builder::new().prefix("temp_file").tempfile_in(&temp_dir_path).unwrap())
//.collect();

////Checks that untracked files are included.
//let mut tidy_flags = TidyFlags::new(&temp_dir_path, &[]);
//tidy_flags.include_untracked = true;

//let mut file_count = 0;
//walk(&temp_dir_path, &tidy_flags, |_, _| false, &mut |_, _| {
//file_count += 1;
//});

//assert!(file_count == 5);

////Checks that untracked files are excluded.
//tidy_flags.include_untracked = false;

//let mut file_count = 0;
//walk(&temp_dir_path, &tidy_flags, |_, _| false, &mut |_, _| {
//file_count += 1;
//});

//assert!(file_count == 0);

////Default behavior include ALL files, including untracked. This is the previous behavior, so any
////downstream functions using `tidy::walk::walk` and now passing in `TidyFlags::default()` should
////be the same as before.
//let tidy_flags = TidyFlags::default();

//let mut file_count = 0;
//walk(&temp_dir_path, &tidy_flags, |_, _| false, &mut |_, _| {
//file_count += 1;
//});

//assert!(file_count == 5);
//}

#[test]
//Checks that the tidy walk function will include files in `tidy/tidy_flags/`.
fn test_tidy_walk_real_files() {
    let test_dir = get_test_dir();
    let tidy_flags = TidyFlags::new(&test_dir, &[]);

    //These files should be tracked and included in `walk`.
    let mut file_count = 0;
    walk(&test_dir, &tidy_flags, |_, _| false, &mut |_, _| {
        file_count += 1;
    });

    //This number could change, but if it does you're already here.
    assert!(file_count == 2);
}
