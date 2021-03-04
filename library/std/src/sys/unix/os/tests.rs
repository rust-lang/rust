use super::*;

#[test]
fn test_glibc_version() {
    // This mostly just tests that the weak linkage doesn't panic wildly...
    glibc_version();
}

#[test]
fn test_parse_glibc_version() {
    let cases = [
        ("0.0", Some((0, 0))),
        ("01.+2", Some((1, 2))),
        ("3.4.5.six", Some((3, 4))),
        ("1", None),
        ("1.-2", None),
        ("1.foo", None),
        ("foo.1", None),
    ];
    for &(version_str, parsed) in cases.iter() {
        assert_eq!(parsed, parse_glibc_version(version_str));
    }
}

#[test]
fn try_all_signals() {
    fn chk(l: &dyn SignalLookupMethod, sig: usize) {
        let got = l.lookup(sig as i32);
        println!("{:2} {:?}", sig, got);
        if let Some(got) = got {
            for &c in got.as_bytes() {
                assert!(c == b' ' || c.is_ascii_graphic(), "sig={} got={:?}", c, &got);
            }
        }
    }

    for sig in 0..NSIG {
        chk(&signal_lookup::descrs, sig);
        chk(&signal_lookup::abbrevs, sig);
    }

    // 1..15 are anciently conventional signal numbers; check they can be looked up:
    for sig in 1..15 {
        assert!(signal_lookup::descrs.lookup(sig).is_some());
    }
}
