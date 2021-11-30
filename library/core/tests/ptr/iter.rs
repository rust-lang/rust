use core::ops::Range;
use core::ptr::IterConst;

#[test]
fn test_ptr_iter_empty() {
    let start = &() as *const () as *const [u8; 100];
    let mut iter = unsafe { IterConst::new(start, start) };
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.nth(0), None);
    assert_eq!(iter.nth(1), None);
    assert_eq!(iter.advance_by(0), Ok(()));
    assert_eq!(iter.advance_by(1), Err(0));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.nth_back(0), None);
    assert_eq!(iter.nth_back(1), None);
    assert_eq!(iter.advance_back_by(0), Ok(()));
    assert_eq!(iter.advance_back_by(1), Err(0));
    assert_eq!(iter, unsafe { IterConst::new(start, start) });
}

#[test]
fn test_ptr_iter() {
    let data = [[0u8; 10]; 12];
    let Range { start, end } = data.as_ptr_range();
    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.size_hint(), (12, Some(12)));
    assert_eq!(iter.next(), Some(&data[0] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[1], end) });
    assert_eq!(iter.nth(0), Some(&data[1] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[2], end) });
    assert_eq!(iter.nth(1), Some(&data[3] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[4], end) });
    assert_eq!(iter.advance_by(0), Ok(()));
    assert_eq!(iter, unsafe { IterConst::new(&data[4], end) });
    assert_eq!(iter.advance_by(1), Ok(()));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], end) });
    assert_eq!(iter.next_back(), Some(&data[11] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], &data[11]) });
    assert_eq!(iter.nth_back(0), Some(&data[10] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], &data[10]) });
    assert_eq!(iter.nth_back(1), Some(&data[8] as _));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], &data[8]) });
    assert_eq!(iter.advance_back_by(0), Ok(()));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], &data[8]) });
    assert_eq!(iter.advance_back_by(1), Ok(()));
    assert_eq!(iter, unsafe { IterConst::new(&data[5], &data[7]) });

    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.nth(20), None);
    assert_eq!(iter, unsafe { IterConst::new(end, end) });

    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.advance_by(20), Err(12));
    assert_eq!(iter, unsafe { IterConst::new(end, end) });

    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.nth_back(20), None);
    assert_eq!(iter, unsafe { IterConst::new(start, start) });

    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.advance_back_by(20), Err(12));
    assert_eq!(iter, unsafe { IterConst::new(start, start) });
}

#[test]
fn test_ptr_iter_zero_sized() {
    let start = &0i32 as *const i32 as *const ();
    let mut iter = unsafe { IterConst::new(start, start) };
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.nth(0), None);
    assert_eq!(iter.nth(1), None);
    assert_eq!(iter.advance_by(0), Ok(()));
    assert_eq!(iter.advance_by(1), Err(0));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.nth_back(0), None);
    assert_eq!(iter.nth_back(1), None);
    assert_eq!(iter.advance_back_by(0), Ok(()));
    assert_eq!(iter.advance_back_by(1), Err(0));
    assert_eq!(iter, unsafe { IterConst::new(start, start) });

    let end = unsafe { start.cast::<i32>().add(1).cast::<()>() };
    let mut iter = unsafe { IterConst::new(start, end) };
    assert_eq!(iter.size_hint(), (usize::MAX, None));
    assert_eq!(iter.next(), Some(start));
    assert_eq!(iter.nth(0), Some(start));
    assert_eq!(iter.nth(1), Some(start));
    assert_eq!(iter.advance_by(0), Ok(()));
    assert_eq!(iter.advance_by(1), Ok(()));
    assert_eq!(iter.next_back(), Some(start));
    assert_eq!(iter.nth_back(0), Some(start));
    assert_eq!(iter.nth_back(1), Some(start));
    assert_eq!(iter.advance_back_by(0), Ok(()));
    assert_eq!(iter.advance_back_by(1), Ok(()));
    assert_eq!(iter, unsafe { IterConst::new(start, end) });
}
