use alloc::io::{Error, ErrorKind, const_error};
use core::{error, fmt};

#[test]
fn test_size() {
    assert!(size_of::<Error>() <= size_of::<[usize; 2]>());
}

#[test]
fn test_debug_error() {
    let code = 6;
    let err = Error::from_raw_os_error(code);
    let mut msg = err.to_string();
    msg.truncate(msg.find('(').unwrap() - 1);
    let kind = err.kind();

    let err = Error::new(ErrorKind::InvalidInput, err);

    let expected = format!(
        "Custom {{ \
         kind: InvalidInput, \
         error: Os {{ \
         code: {:?}, \
         kind: {:?}, \
         message: {:?} \
         }} \
         }}",
        code, kind, msg
    );
    assert_eq!(format!("{err:?}"), expected);

    let err = Error::from(ErrorKind::AddrInUse);
    assert_eq!(format!("{err:?}"), "Kind(AddrInUse)");

    let err = Error::READ_EXACT_EOF;
    assert_eq!(
        format!("{err:?}"),
        "Error { kind: UnexpectedEof, message: \"failed to fill whole buffer\" }"
    );
}

#[test]
fn test_downcasting() {
    #[derive(Debug)]
    struct TestError;

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("asdf")
        }
    }

    impl error::Error for TestError {}

    // we have to call all of these UFCS style right now since method
    // resolution won't implicitly drop the Send+Sync bounds
    let mut err = Error::new(ErrorKind::Other, TestError);
    assert!(err.get_ref().unwrap().is::<TestError>());
    assert_eq!("asdf", err.get_ref().unwrap().to_string());
    assert!(err.get_mut().unwrap().is::<TestError>());
    let extracted = err.into_inner().unwrap();
    extracted.downcast::<TestError>().unwrap();
}

#[test]
fn test_const() {
    const E: Error = const_error!(ErrorKind::NotFound, "hello");

    assert_eq!(E.kind(), ErrorKind::NotFound);
    assert_eq!(E.to_string(), "hello");
    assert!(format!("{E:?}").contains("\"hello\""));
    assert!(format!("{E:?}").contains("NotFound"));
}

#[test]
fn test_os_packing() {
    for code in -20..20 {
        let e = Error::from_raw_os_error(code);
        assert_eq!(e.raw_os_error(), Some(code));
    }
}

#[test]
fn test_errorkind_packing() {
    assert_eq!(Error::from(ErrorKind::NotFound).kind(), ErrorKind::NotFound);
    assert_eq!(Error::from(ErrorKind::PermissionDenied).kind(), ErrorKind::PermissionDenied);
    assert_eq!(Error::from(ErrorKind::Uncategorized).kind(), ErrorKind::Uncategorized);
}

#[test]
fn test_simple_message_packing() {
    use std::io::ErrorKind::*;
    macro_rules! check_simple_msg {
        ($err:expr, $kind:ident, $msg:literal) => {{
            let e = &$err;
            // Check that the public api is right.
            assert_eq!(e.kind(), $kind);
            assert!(format!("{e:?}").contains($msg));
            assert!(format!("{e}").contains($msg));
        }};
    }

    let not_static = const_error!(Uncategorized, "not a constant!");
    check_simple_msg!(not_static, Uncategorized, "not a constant!");

    const CONST: Error = const_error!(NotFound, "definitely a constant!");
    check_simple_msg!(CONST, NotFound, "definitely a constant!");

    static STATIC: Error = const_error!(BrokenPipe, "a constant, sort of!");
    check_simple_msg!(STATIC, BrokenPipe, "a constant, sort of!");
}

#[derive(Debug, PartialEq)]
struct Bojji(bool);
impl error::Error for Bojji {}
impl fmt::Display for Bojji {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ah! {:?}", self)
    }
}

#[derive(Debug)]
struct E;

impl fmt::Display for E {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl error::Error for E {}

#[test]
fn test_std_io_error_downcast() {
    // Case 1: custom error, downcast succeeds
    let io_error = Error::new(ErrorKind::Other, Bojji(true));
    let e: Bojji = io_error.downcast().unwrap();
    assert!(e.0);

    // Case 2: custom error, downcast fails
    let io_error = Error::new(ErrorKind::Other, Bojji(true));
    let io_error = io_error.downcast::<E>().unwrap_err();

    //   ensures that the custom error is intact
    assert_eq!(ErrorKind::Other, io_error.kind());
    let e: Bojji = io_error.downcast().unwrap();
    assert!(e.0);

    // Case 3: os error
    let errno = 20;
    let io_error = Error::from_raw_os_error(errno);
    let io_error = io_error.downcast::<E>().unwrap_err();

    assert_eq!(errno, io_error.raw_os_error().unwrap());

    // Case 4: simple
    let kind = ErrorKind::OutOfMemory;
    let io_error: Error = kind.into();
    let io_error = io_error.downcast::<E>().unwrap_err();

    assert_eq!(kind, io_error.kind());

    // Case 5: simple message
    let io_error = const_error!(ErrorKind::Other, "simple message error test");
    let io_error = io_error.downcast::<E>().unwrap_err();

    assert_eq!(io_error.kind(), ErrorKind::Other);
    assert_eq!(format!("{io_error}"), "simple message error test");
}
