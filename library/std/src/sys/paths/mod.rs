cfg_select! {
    target_os = "hermit" => {
        mod hermit;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::hermit::{getcwd, temp_dir};
            pub use super::unsupported::{
                JoinPathsError, SplitPaths, chdir, current_exe, home_dir, join_paths, split_paths,
            };
        }
    }
    target_os = "motor" => {
        mod motor;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::motor::{
                JoinPathsError, SplitPaths, chdir, current_exe, getcwd, home_dir, join_paths,
                split_paths, temp_dir,
            };
        }
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        #[expect(dead_code)]
        mod unsupported;
        mod imp {
            pub use super::sgx::chdir;
            pub use super::unsupported::{
                JoinPathsError, SplitPaths, current_exe, getcwd, home_dir, join_paths, split_paths,
                temp_dir,
            };
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
            pub use super::unsupported::{
                JoinPathsError, SplitPaths, current_exe, home_dir, join_paths, split_paths,
            };
            pub use super::wasi::{chdir, getcwd, temp_dir};
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

pub(crate) use imp::{
    JoinPathsError, SplitPaths, chdir, current_exe, getcwd, home_dir, join_paths, split_paths,
    temp_dir,
};
