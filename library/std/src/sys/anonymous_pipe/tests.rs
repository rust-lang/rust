use super::*;
use crate::io::{Read, Write};

#[test]
fn pipe_creation_and_rw() {
    let (mut rx, mut tx) = pipe().unwrap();
    tx.write_all(b"12345").unwrap();
    drop(tx);

    let mut s = String::new();
    rx.read_to_string(&mut s).unwrap();
    assert_eq!(s, "12345");
}

#[test]
fn pipe_try_clone_and_rw() {
    let (mut rx, mut tx) = pipe().unwrap();
    tx.try_clone().unwrap().write_all(b"12").unwrap();
    tx.write_all(b"345").unwrap();
    drop(tx);

    let mut s = String::new();
    rx.try_clone().unwrap().take(3).read_to_string(&mut s).unwrap();
    assert_eq!(s, "123");

    s.clear();
    rx.read_to_string(&mut s).unwrap();
    assert_eq!(s, "45");
}

#[cfg(unix)]
mod unix_specific {
    use super::*;

    use crate::{
        fs::File,
        io,
        os::fd::{AsRawFd, OwnedFd},
    };

    #[test]
    fn pipe_owned_fd_round_trip_conversion() {
        let (rx, tx) = pipe().unwrap();
        let raw_fds = (rx.as_raw_fd(), tx.as_raw_fd());
        let (rx_owned_fd, tx_owned_fd) = (OwnedFd::from(rx), OwnedFd::from(tx));

        let rx = PipeReader::try_from(rx_owned_fd).unwrap();
        let tx = PipeWriter::try_from(tx_owned_fd).unwrap();
        assert_eq!(raw_fds, (rx.as_raw_fd(), tx.as_raw_fd()));
    }

    #[test]
    fn convert_from_non_pipe_to_pipe_reader_shall_fail() {
        let file = File::open("/dev/zero").unwrap();
        let err = PipeReader::try_from(OwnedFd::from(file)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), "Not a pipe");
    }

    #[test]
    fn convert_from_non_pipe_to_pipe_writer_shall_fail() {
        let file = File::options().write(true).open("/dev/null").unwrap();
        let err = PipeWriter::try_from(OwnedFd::from(file)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), "Not a pipe");
    }

    #[test]
    fn convert_pipe_writer_to_pipe_reader_shall_fail() {
        let (_, tx) = pipe().unwrap();
        let fd = tx.as_raw_fd();
        let err = PipeReader::try_from(OwnedFd::from(tx)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), format!("Pipe {fd} is not readable"));
    }

    #[test]
    fn convert_pipe_reader_to_pipe_writer_shall_fail() {
        let (rx, _) = pipe().unwrap();
        let fd = rx.as_raw_fd();
        let err = PipeWriter::try_from(OwnedFd::from(rx)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), format!("Pipe {fd} is not writable"));
    }
}

#[cfg(windows)]
mod windows_specific {
    use super::*;

    use crate::{
        io,
        os::windows::io::{AsHandle, AsRawHandle, OwnedHandle},
    };

    #[test]
    fn pipe_owned_handle_round_trip_conversion() {
        let (rx, tx) = pipe().unwrap();
        let raw_handles = (rx.as_raw_handle(), tx.as_raw_handle());
        let (rx_owned_handle, tx_owned_handle) = (OwnedHandle::from(rx), OwnedHandle::from(tx));

        let rx = PipeReader::try_from(rx_owned_handle).unwrap();
        let tx = PipeWriter::try_from(tx_owned_handle).unwrap();
        assert_eq!(raw_handles, (rx.as_raw_handle(), tx.as_raw_handle()));
    }

    #[test]
    fn convert_from_non_pipe_to_pipe_reader_shall_fail() {
        let file = io::stdin().as_handle().try_clone_to_owned().unwrap();
        let err = PipeReader::try_from(OwnedHandle::from(file)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), "Not a pipe");
    }

    #[test]
    fn convert_from_non_pipe_to_pipe_writer_shall_fail() {
        let file = io::stdout().as_handle().try_clone_to_owned().unwrap();
        let err = PipeWriter::try_from(OwnedHandle::from(file)).unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(format!("{}", err.get_ref().unwrap()), "Not a pipe");
    }
}
