// cdb-only
//@ min-cdb-version: 10.0.18317.1001
//@ compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g

// cdb-command: dx x,d
// cdb-check:x,d              : Ok [Type: enum2$<core::result::Result<i32,ref$<str$> > >]
// cdb-check:    [...] __0              : -3 [Type: int]

// cdb-command: dx y
// cdb-check:y                : Err [Type: enum2$<core::result::Result<i32,ref$<str$> > >]
// cdb-check:    [...] __0              : "Some error message" [Type: ref$<str$>]

fn main() {
    let x: Result<i32, &str> = Ok(-3);
    assert_eq!(x.is_ok(), true);

    let y: Result<i32, &str> = Err("Some error message");
    assert_eq!(y.is_ok(), false);

    zzz(); // #break.
}

fn zzz() {
    ()
}
