use super::{Error, ErrorKind};
use crate::sys::decode_error_kind;
use crate::sys::os::error_string;

#[test]
fn test_debug_error() {
    let code = 6;
    let msg = error_string(code);
    let kind = decode_error_kind(code);
    let err = Error::new(ErrorKind::InvalidInput, Error::from_raw_os_error(code));
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
}
