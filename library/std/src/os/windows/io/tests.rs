#[test]
fn test_niche_optimizations_socket() {
    use crate::mem::size_of;
    use crate::os::windows::io::{
        BorrowedSocket, FromRawSocket, IntoRawSocket, OwnedSocket, RawSocket,
    };

    assert_eq!(size_of::<Option<OwnedSocket>>(), size_of::<RawSocket>());
    assert_eq!(size_of::<Option<BorrowedSocket<'static>>>(), size_of::<RawSocket>(),);
    unsafe {
        assert_eq!(OwnedSocket::from_raw_socket(RawSocket::MIN).into_raw_socket(), RawSocket::MIN);
        assert_eq!(OwnedSocket::from_raw_socket(RawSocket::MAX).into_raw_socket(), RawSocket::MAX);
        assert_eq!(
            Some(OwnedSocket::from_raw_socket(RawSocket::MIN)).unwrap().into_raw_socket(),
            RawSocket::MIN
        );
        assert_eq!(
            Some(OwnedSocket::from_raw_socket(RawSocket::MAX)).unwrap().into_raw_socket(),
            RawSocket::MAX
        );
    }
}
