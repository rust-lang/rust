cfg_select! {
    target_os = "hermit" => {
        mod hermit;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::hermit::{getcwd, temp_dir};
            pub use super::unsupported::{chdir, SplitPaths, split_paths, JoinPathsError, join_paths, current_exe, home_dir};
        }
    }
    target_os = "motor" => {
        mod motor;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::motor::{getcwd, chdir, current_exe, temp_dir};
            pub use super::unsupported::{SplitPaths, split_paths, JoinPathsError, join_paths, home_dir};
        }
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::sgx::chdir;
            pub use super::unsupported::{getcwd, SplitPaths, split_paths, JoinPathsError, join_paths, current_exe, temp_dir, home_dir};
        }
    }
    target_os = "uefi" => {
        mod uefi;
        use uefi as imp;
    }
    target_family = "unix" => {
        mod unix;
        use unix as imp;
    }
    target_os = "wasi" => {
        mod wasi;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::wasi::{getcwd, chdir, temp_dir};
            pub use super::unsupported::{current_exe, SplitPaths, split_paths, JoinPathsError, join_paths, home_dir};
        }
    }
    target_os = "windows" => {
        mod windows;
        use windows as imp;
    }
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

pub use imp::{
    JoinPathsError, SplitPaths, chdir, current_exe, getcwd, home_dir, join_paths, split_paths,
    temp_dir,
};
