//@ rustfmt-unstable: true

// Original issue
fn main() {
    let a = cfg_select! {
        feature = "foo" => {{
            let bar = "foo";
            bar.to_owned()
        }}
        _ => String::new(),
    };

    let with_trailing_double_block_comma = cfg_select! {
        feature = "foo" => {{
            let bar = "foo";
            bar.to_owned()
        }},
        _ => String::new(),
    };

    println!("{a}");
}

// Issues miri ran into that required  #[rustfmt::skip] to work around
fn miri_issues() {
    let result = cfg_select! {
        unix => fs::remove_file(path),
        windows => {{
            // This should be able to remove symlinks to dirs, but on Windows that requires
            // `remove_dir`. Our work-around is racy but there's not a lot we can do about that.
            // FIXME: maybe we can retry based on the error code?
            use std::os::windows::fs::FileTypeExt;
            let metadata = path.symlink_metadata();
            if metadata.is_ok_and(|m| m.file_type().is_symlink_dir()) {
                fs::remove_dir(path)
            } else {
                fs::remove_file(path)
            }
        }}
    };

    let result = cfg_select! {
        unix => fs::remove_dir(path),
        windows => {{
            // This should *not* be able to remove symlinks to dirs, but on Windows it is. Our
            // work-around is racy (it might change to being a directory symlink after we
            // checked it), but there's not a lot we can do about that.
            use std::os::windows::fs::FileTypeExt;
            let metadata = path.symlink_metadata();
            if metadata.is_ok_and(|m| m.file_type().is_symlink_dir()) {
                return this.set_errno_and_return_neg1_i32(LibcError("ENOTDIR"));
            } else {
                fs::remove_dir(path)
            }
        }}
    };

    let listener = cfg_select! {
        // Turn the `socket2::Socket` into a `mio::TcpListener`.
        // We follow Tokio's TCP socket implementation:
        // See <https://github.com/tokio-rs/tokio/blob/2120bee/tokio/src/net/tcp/socket.rs#L906-L925>
        // SAFETY: mio specifies that it is safe to use `from_raw_fd` and
        // `from_raw_socket` as long as it's ensured that the socket is non-blocking.
        // Because we immediately make the socket non-blocking after creation, that
        // is always guaranteed.
        unix => {{
            use std::os::fd::{FromRawFd, IntoRawFd};
            let raw_fd = socket.into_raw_fd();
            unsafe { TcpListener::from_raw_fd(raw_fd) }
        }},
        windows => {{
            use std::os::windows::io::{FromRawSocket, IntoRawSocket};
            let raw_socket = socket.into_raw_socket();
            unsafe { TcpListener::from_raw_socket(raw_socket) }
        }},
        _ => unreachable!("unsupported host platform"),
    };

    let stream = cfg_select! {
        // Turn the `socket2::Socket` into a `mio::TcpStream`.
        // We follow Tokio's TCP socket implementation:
        // See <https://github.com/tokio-rs/tokio/blob/2120bee/tokio/src/net/tcp/socket.rs#L841-L869>
        // SAFETY: mio specifies that it is safe to use `from_raw_fd` and
        // `from_raw_socket` as long as it's ensured that the socket is non-blocking.
        // Because we immediately make the socket non-blocking after creation, that
        // is always guaranteed.
        unix => {{
            use std::os::fd::{FromRawFd, IntoRawFd};
            let raw_fd = socket.into_raw_fd();
            unsafe { TcpStream::from_raw_fd(raw_fd) }
        }},
        windows => {{
            use std::os::windows::io::{FromRawSocket, IntoRawSocket};
            let raw_socket = socket.into_raw_socket();
            unsafe { TcpStream::from_raw_socket(raw_socket) }
        }},
        _ => unreachable!("unsupported host platform"),
    };
}
