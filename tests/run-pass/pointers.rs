fn one_line_ref() -> i16 {
    *&1
}

fn basic_ref() -> i16 {
    let x = &1;
    *x
}

fn basic_ref_mut() -> i16 {
    let x = &mut 1;
    *x += 2;
    *x
}

fn basic_ref_mut_var() -> i16 {
    let mut a = 1;
    {
        let x = &mut a;
        *x += 2;
    }
    a
}

fn tuple_ref_mut() -> (i8, i8) {
    let mut t = (10, 20);
    {
        let x = &mut t.1;
        *x += 2;
    }
    t
}

fn match_ref_mut() -> i8 {
    let mut t = (20, 22);
    {
        let opt = Some(&mut t);
        match opt {
            Some(&mut (ref mut x, ref mut y)) => *x += *y,
            None => {},
        }
    }
    t.0
}

fn dangling_pointer() -> *const i32 {
    let b = Box::new((42, 42)); // make it bigger than the alignment, so that there is some "room" after this pointer
    &b.0 as *const i32
}

fn main() {
    assert_eq!(one_line_ref(), 1);
    assert_eq!(basic_ref(), 1);
    assert_eq!(basic_ref_mut(), 3);
    assert_eq!(basic_ref_mut_var(), 3);
    assert_eq!(tuple_ref_mut(), (10, 22));
    assert_eq!(match_ref_mut(), 42);

    // Compare even dangling pointers with NULL, and with others in the same allocation, including
    // out-of-bounds.
    assert!(dangling_pointer() != std::ptr::null());
    assert!(match dangling_pointer() as usize { 0 => false, _ => true });
    let dangling = dangling_pointer();
    assert!(dangling == dangling);
    assert!(dangling.wrapping_add(1) != dangling);
    assert!(dangling.wrapping_sub(1) != dangling);

    // Compare pointer with BIG integers
    let dangling = dangling as usize;
    assert!(dangling != usize::MAX);
    assert!(dangling != usize::MAX - 1);
    assert!(dangling != usize::MAX - 2);
    assert!(dangling != usize::MAX - 3); // this is even 4-aligned, but it still cannot be equal because of the extra "room" after this pointer
    assert_eq!((usize::MAX - 3) % 4, 0); // just to be sure we got this right

    // Compare pointer with unaligned integers
    assert!(dangling != 1usize);
    assert!(dangling != 2usize);
    assert!(dangling != 3usize);
    // 4 is a possible choice! So we cannot compare with that.
    assert!(dangling != 5usize);
    assert!(dangling != 6usize);
    assert!(dangling != 7usize);

    // Using inequality to do the comparison.
    assert!(dangling > 0);
    assert!(dangling > 1);
    assert!(dangling > 2);
    assert!(dangling > 3);
    assert!(dangling >= 4);
}
