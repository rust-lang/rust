pub const DEFAULT_BUF_SIZE: usize = 8 * 1024;

#[cfg(test)]
#[allow(dead_code)] // not used on emscripten
pub mod test {
    use crate::path::{Path, PathBuf};
    use crate::env;
    use crate::fs;
    use rand::RngCore;

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
            fs::remove_dir_all(p).unwrap();
        }
    }

    pub fn tmpdir() -> TempDir {
        let p = env::temp_dir();
        let mut r = rand::thread_rng();
        let ret = p.join(&format!("rust-{}", r.next_u32()));
        fs::create_dir(&ret).unwrap();
        TempDir(ret)
    }
}
