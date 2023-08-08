use std::process::ExitCode;

use build_helper::git::get_git_modified_files;
use suggest_tests::get_suggestions;

fn main() -> ExitCode {
    let modified_files = get_git_modified_files(None, &Vec::new());
    let modified_files = match modified_files {
        Ok(Some(files)) => files,
        Ok(None) => {
            eprintln!("git error");
            return ExitCode::FAILURE;
        }
        Err(err) => {
            eprintln!("Could not get modified files from git: \"{err}\"");
            return ExitCode::FAILURE;
        }
    };

    let suggestions = get_suggestions(&modified_files);

    for sug in &suggestions {
        println!("{sug}");
    }

    ExitCode::SUCCESS
}
