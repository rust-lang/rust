// Bare metal platforms usually have very small amounts of RAM
// (in the order of hundreds of KB)
pub const DEFAULT_BUF_SIZE: usize = if cfg!(target_os = "espidf") { 512 } else { 8 * 1024 };

#[cfg(test)]
#[allow(dead_code)] // not used on emscripten and wasi
pub mod test {
    use rand::RngCore;

    use crate::path::{Path, PathBuf};
    use crate::{env, fs, thread};

    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn join(&self, path: &str) -> PathBuf {
            let TempDir(ref p) = *self;
            p.join(path)
        }

        pub fn path(&self) -> &Path {
            let TempDir(ref p) = *self;
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Gee, seeing how we're testing the fs module I sure hope that we
            // at least implement this correctly!
            let TempDir(ref p) = *self;
            let result = fs::remove_dir_all(p);
            // Avoid panicking while panicking as this causes the process to
            // immediately abort, without displaying test results.
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }

    #[track_caller] // for `test_rng`
    pub fn tmpdir() -> TempDir {
        let p = env::temp_dir();
        let mut r = crate::test_helpers::test_rng();
        let ret = p.join(&format!("rust-{}", r.next_u32()));
        fs::create_dir(&ret).unwrap();
        TempDir(ret)
    }
}
