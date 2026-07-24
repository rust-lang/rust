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

// #115325: on Apple, `send` rejects a length > `c_int::MAX` with `EINVAL`, so
// the clamp must not regress to the unbounded `wrlen_t::MAX`.
#[test]
fn max_send_len_within_platform_limit() {
    if cfg!(target_vendor = "apple") {
        assert_eq!(MAX_SEND_LEN, c_int::MAX as usize);
    } else {
        assert_eq!(MAX_SEND_LEN, <wrlen_t>::MAX as usize);
    }
    assert_eq!(crate::cmp::min(MAX_SEND_LEN.saturating_add(1), MAX_SEND_LEN), MAX_SEND_LEN);
}
