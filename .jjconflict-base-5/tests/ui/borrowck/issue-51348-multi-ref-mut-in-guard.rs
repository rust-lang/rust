// We used to ICE if you had a single match arm with multiple
// candidate patterns with `ref mut` identifiers used in the arm's
// guard.
//
// Also, this test expands on the original bug's example by actually
// trying to double check that we are matching against the right part
// of the input data based on which candidate pattern actually fired.

//@ run-pass

fn foo(x: &mut Result<(u32, u32), (u32, u32)>) -> u32 {
    match *x {
        Ok((ref mut v, _)) | Err((_, ref mut v)) if *v > 0 => { *v }
        _ => { 0 }
    }
}

fn main() {
    assert_eq!(foo(&mut Ok((3, 4))), 3);
    assert_eq!(foo(&mut Err((3, 4))), 4);
}
