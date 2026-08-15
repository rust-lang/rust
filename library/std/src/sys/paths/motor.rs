use crate::io;
use crate::os::motor::ffi::OsStrExt;
use crate::path::{self, PathBuf};
use crate::sys::pal::map_motor_error;

pub fn getcwd() -> io::Result<PathBuf> {
    moto_rt::fs::getcwd().map(PathBuf::from).map_err(map_motor_error)
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    moto_rt::fs::chdir(path.as_os_str().as_str()).map_err(map_motor_error)
}

pub fn current_exe() -> io::Result<PathBuf> {
    moto_rt::process::current_exe().map(PathBuf::from).map_err(map_motor_error)
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from(moto_rt::fs::TEMP_DIR)
}
