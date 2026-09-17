use super::*;
use crate::collections::HashMap;

#[test]
fn no_lookup_host_duplicates() {
    let mut addrs = HashMap::new();
    let lh = match lookup_host("localhost", 0) {
        Ok(lh) => lh,
        Err(e) => panic!("couldn't resolve `localhost`: {e}"),
    };
    for sa in lh {
        *addrs.entry(sa).or_insert(0) += 1;
    }
    assert_eq!(
        addrs.iter().filter(|&(_, &v)| v > 1).collect::<Vec<_>>(),
        vec![],
        "There should be no duplicate localhost entries"
    );
}

// On non-Windows platforms, the maximum valid length to pass into `send` is
// not the same as the maximum of the *type* used for the length. Ensure that
// the proper maximum is used, and that we do not regress to simply clamping to
// `wrlen_t::MAX`
//
// On Apple (per 115325), `send` takes a `size_t` length, but rejects any
// length > `c_int::MAX` with `EINVAL`.
//
// On QNX, `send` with length > `c_int::MAX` returns an incorrect count of bytes
// written.
//
// On Windows, `send` takes an `i32` length and returns an `i32`.
//
// On other platforms, `send` takes a `size_t` and returns an `ssize_t`, so
// sends larger then `ssize_t::MAX` will (maybe silently) return bad lengths.
#[test]
fn max_send_len_within_platform_limit() {
    if cfg!(any(target_vendor = "apple", target_os = "nto", target_os = "qnx")) {
        assert_eq!(MAX_SEND_LEN, c_int::MAX as usize);
    } else if cfg!(target_os = "windows") {
        assert_eq!(MAX_SEND_LEN, i32::MAX as usize);
    } else {
        assert_eq!(MAX_SEND_LEN, libc::ssize_t::MAX as usize);
    }
}
