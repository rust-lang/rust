//@ rustfmt-unstable: true
//@ rustfmt-edition: 2024

fn gettid() -> u64 {
    cfg_select! {
        any(target_os = "solaris", target_os = "illumos") => {unsafe {
            libc::pthread_self() as u64
        }}
    }
}
