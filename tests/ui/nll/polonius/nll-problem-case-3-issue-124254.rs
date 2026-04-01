// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #124254
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

fn find_lowest_or_first_empty_pos(list: &mut [Option<u8>]) -> &mut Option<u8> {
    let mut low_pos_val: Option<(usize, u8)> = None;
    for (idx, i) in list.iter_mut().enumerate() {
        let Some(s) = i else {
            return i;
        };

        low_pos_val = match low_pos_val {
            Some((_oidx, oval)) if oval > *s => Some((idx, *s)),
            Some(old) => Some(old),
            None => Some((idx, *s)),
        };
    }
    let Some((lowest_idx, _)) = low_pos_val else {
        unreachable!("Can't have zero length list!");
    };
    &mut list[lowest_idx]
}

fn main() {
    let mut list = [Some(1), Some(2), None, Some(3)];
    let v = find_lowest_or_first_empty_pos(&mut list);
    assert!(v.is_none());
    assert_eq!(v as *mut _ as usize, list.as_ptr().wrapping_add(2) as usize);

    let mut list = [Some(1), Some(2), Some(3), Some(0)];
    let v = find_lowest_or_first_empty_pos(&mut list);
    assert_eq!(v, &mut Some(0));
    assert_eq!(v as *mut _ as usize, list.as_ptr().wrapping_add(3) as usize);

    println!("pass");
}
