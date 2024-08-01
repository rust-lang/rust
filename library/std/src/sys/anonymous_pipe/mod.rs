cfg_if::cfg_if! {
    if #[cfg(unix)] {
        mod unix;
        pub(crate) use unix::{AnonPipe, pipe};

        #[cfg(all(test, not(miri)))]
        mod tests;
    } else if #[cfg(windows)] {
        mod windows;
        pub(crate) use windows::{AnonPipe, pipe};

        #[cfg(all(test, not(miri)))]
        mod tests;
    } else {
        mod unsupported;
        pub(crate) use unsupported::{AnonPipe, pipe};
    }
}
