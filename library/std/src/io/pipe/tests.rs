use crate::io::{Read, Write, pipe};

#[test]
#[cfg(all(windows, unix, not(miri)))]
fn pipe_creation_clone_and_rw() {
    let (rx, tx) = pipe().unwrap();

    tx.try_clone().unwrap().write_all(b"12345").unwrap();
    drop(tx);

    let mut rx2 = rx.try_clone().unwrap();
    drop(rx);

    let mut s = String::new();
    rx2.read_to_string(&mut s).unwrap();
    drop(rx2);
    assert_eq!(s, "12345");
}
