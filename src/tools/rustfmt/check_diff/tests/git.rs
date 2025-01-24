use check_diff::clone_git_repo;

use tempfile::Builder;

#[test]
fn clone_repo_test() {
    // Creates an empty directory in the current working directory
    let dir = Builder::new().tempdir_in("").unwrap();
    let sample_repo = "https://github.com/rust-lang/rustfmt.git";
    let dest_path = dir.path();
    let result = clone_git_repo(sample_repo, dest_path);
    assert!(result.is_ok());
    // check whether a .git folder exists after cloning the repo
    let git_repo = dest_path.join(".git");
    assert!(git_repo.exists());
}
