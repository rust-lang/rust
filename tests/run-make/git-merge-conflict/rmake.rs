// Checks the output in cases where a manual `git merge` had conflicts.

use run_make_support::{diff, rfs, rustc};

fn main() {
    // We use `.rust` purely to avoid `fmt` check on a file with conflict markers
    rfs::copy_dir_all("git", ".git");
    let file_out = rustc().input("main.rust").run_fail().stderr_utf8();
    diff().expected_file("file.stderr").actual_text("actual-file-stderr", file_out).run();
}
