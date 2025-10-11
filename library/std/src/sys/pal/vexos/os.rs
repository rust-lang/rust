#[expect(dead_code)]
#[path = "../unsupported/os.rs"]
mod unsupported_os;
pub use unsupported_os::{
    JoinPathsError, SplitPaths, chdir, current_exe, errno, error_string, getcwd, getpid, home_dir,
    join_paths, split_paths, temp_dir,
};

pub use super::unsupported;

pub fn exit(_code: i32) -> ! {
    unsafe {
        vex_sdk::vexSystemExitRequest();

        loop {
            vex_sdk::vexTasksRun();
        }
    }
}
