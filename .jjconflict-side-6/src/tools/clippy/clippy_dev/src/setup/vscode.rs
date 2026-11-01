use std::fs;
use std::path::Path;

const VSCODE_DIR: &str = ".vscode";
const TASK_SOURCE_FILE: &str = "util/etc/vscode-tasks.json";
const TASK_TARGET_FILE: &str = ".vscode/tasks.json";

pub fn install_tasks(force_override: bool) {
    if !check_install_precondition(force_override) {
        return;
    }

    match fs::copy(TASK_SOURCE_FILE, TASK_TARGET_FILE) {
        Ok(_) => {
            println!("info: the task file can be removed with `cargo dev remove vscode-tasks`");
            println!("vscode tasks successfully installed");
        },
        Err(err) => eprintln!("error: unable to copy `{TASK_SOURCE_FILE}` to `{TASK_TARGET_FILE}` ({err})"),
    }
}

fn check_install_precondition(force_override: bool) -> bool {
    let vs_dir_path = Path::new(VSCODE_DIR);
    if vs_dir_path.exists() {
        // verify the target will be valid
        if !vs_dir_path.is_dir() {
            eprintln!("error: the `.vscode` path exists but seems to be a file");
            return false;
        }

        // make sure that we don't override any existing tasks by accident
        let path = Path::new(TASK_TARGET_FILE);
        if path.exists() {
            if force_override {
                return delete_vs_task_file(path);
            }

            eprintln!("error: there is already a `task.json` file inside the `{VSCODE_DIR}` directory");
            println!("info: use the `--force-override` flag to override the existing `task.json` file");
            return false;
        }
    } else {
        match fs::create_dir(vs_dir_path) {
            Ok(()) => {
                println!("info: created `{VSCODE_DIR}` directory for clippy");
            },
            Err(err) => {
                eprintln!("error: the task target directory `{VSCODE_DIR}` could not be created ({err})");
            },
        }
    }

    true
}

pub fn remove_tasks() {
    let path = Path::new(TASK_TARGET_FILE);
    if path.exists() {
        if delete_vs_task_file(path) {
            try_delete_vs_directory_if_empty();
            println!("vscode tasks successfully removed");
        }
    } else {
        println!("no vscode tasks were found");
    }
}

fn delete_vs_task_file(path: &Path) -> bool {
    if let Err(err) = fs::remove_file(path) {
        eprintln!("error: unable to delete the existing `tasks.json` file ({err})");
        return false;
    }

    true
}

/// This function will try to delete the `.vscode` directory if it's empty.
/// It may fail silently.
fn try_delete_vs_directory_if_empty() {
    let path = Path::new(VSCODE_DIR);
    if path.read_dir().is_ok_and(|mut iter| iter.next().is_none()) {
        // The directory is empty. We just try to delete it but allow a silence
        // fail as an empty `.vscode` directory is still valid
        let _silence_result = fs::remove_dir(path);
    } else {
        // The directory is not empty or could not be read. Either way don't take
        // any further actions
    }
}
