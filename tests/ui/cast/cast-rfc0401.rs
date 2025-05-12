//@ run-pass

#![allow(dead_code)]

use std::vec;

enum Simple {
    A,
    B,
    C
}

enum Valued {
    H8=163,
    Z=0,
    X=256,
    H7=67,
}

enum ValuedSigned {
    M1=-1,
    P1=1
}

fn main()
{
    // coercion-cast
    let mut it = vec![137].into_iter();
    let itr: &mut vec::IntoIter<u32> = &mut it;
    assert_eq!((itr as &mut dyn Iterator<Item=u32>).next(), Some(137));
    assert_eq!((itr as &mut dyn Iterator<Item=u32>).next(), None);

    assert_eq!(Some(4u32) as Option<u32>, Some(4u32));
    assert_eq!((1u32,2u32) as (u32,u32), (1,2));

    // this isn't prim-int-cast. Check that it works.
    assert_eq!(false as bool, false);
    assert_eq!(true as bool, true);

    // numeric-cast
    let l: u64 = 0x8090a0b0c0d0e0f0;
    let lsz: usize = l as usize;
    assert_eq!(l as u32, 0xc0d0e0f0);

    // numeric-cast
    assert_eq!(l as u8, 0xf0);
    assert_eq!(l as i8,-0x10);
    assert_eq!(l as u32, 0xc0d0e0f0);
    assert_eq!(l as u32 as usize as u32, l as u32);
    assert_eq!(l as i32,-0x3f2f1f10);
    assert_eq!(l as i32 as isize as i32, l as i32);
    assert_eq!(l as i64,-0x7f6f5f4f3f2f1f10);

    assert_eq!(0 as f64, 0f64);
    assert_eq!(1 as f64, 1f64);

    assert_eq!(l as f64, 9264081114510712022f64);

    assert_eq!(l as i64 as f64, -9182662959198838444f64);
//  float overflow : needs fixing
//  assert_eq!(l as f32 as i64 as u64, 9264082620822882088u64);
//  assert_eq!(l as i64 as f32 as i64, 9182664080220408446i64);

    assert_eq!(4294967040f32 as u32, 0xffffff00u32);
    assert_eq!(1.844674407370955e19f64 as u64, 0xfffffffffffff800u64);

    assert_eq!(9.223372036854775e18f64 as i64, 0x7ffffffffffffc00i64);
    assert_eq!(-9.223372036854776e18f64 as i64, 0x8000000000000000u64 as i64);

    // addr-ptr-cast/ptr-addr-cast (thin ptr)
    let p: *const [u8; 1] = lsz as *const [u8; 1];
    assert_eq!(p as usize, lsz);

    // ptr-ptr-cast (thin ptr)
    let w: *const () = p as *const ();
    assert_eq!(w as usize, lsz);

    // ptr-ptr-cast (fat->thin)
    let u: *const [u8] = unsafe{&*p};
    assert_eq!(u as *const u8, p as *const u8);
    assert_eq!(u as *const u16, p as *const u16);

    // ptr-ptr-cast (Length vtables)
    let mut l : [u16; 2] = [0,1];
    let w: *mut [u8; 2] = &mut l as *mut [u16; 2] as *mut _;
    let w: *mut [u8] = unsafe {&mut *w};
    let w_u16 : *const [u16] = w as *const [u16];
    assert_eq!(unsafe{&*w_u16}, &l);

    let s: *mut str = w as *mut str;
    let l_via_str = unsafe{&*(s as *const [u16])};
    assert_eq!(&l, l_via_str);

    // ptr-ptr-cast (Length vtables, check length is preserved)
    let l: [[u8; 3]; 2] = [[3, 2, 6], [4, 5, 1]];
    let p: *const [[u8; 3]] = &l;
    let p: &[[u8; 2]] = unsafe {&*(p as *const [[u8; 2]])};
    assert_eq!(p, [[3, 2], [6, 4]]);

    // enum-cast
    assert_eq!(Simple::A as u8, 0);
    assert_eq!(Simple::B as u8, 1);

    assert_eq!(Valued::H8 as i8, -93);
    assert_eq!(Valued::H7 as i8, 67);
    assert_eq!(Valued::Z as i8, 0);

    assert_eq!(Valued::H8 as u8, 163);
    assert_eq!(Valued::H7 as u8, 67);
    assert_eq!(Valued::Z as u8, 0);

    assert_eq!(Valued::H8 as u16, 163);
    assert_eq!(Valued::Z as u16, 0);
    assert_eq!(Valued::H8 as u16, 163);
    assert_eq!(Valued::Z as u16, 0);

    assert_eq!(ValuedSigned::M1 as u16, 65535);
    assert_eq!(ValuedSigned::M1 as i16, -1);
    assert_eq!(ValuedSigned::P1 as u16, 1);
    assert_eq!(ValuedSigned::P1 as i16, 1);

    // prim-int-cast
    assert_eq!(false as u16, 0);
    assert_eq!(true as u16, 1);
    assert_eq!(false as i64, 0);
    assert_eq!(true as i64, 1);
    assert_eq!('a' as u32, 0x61);
    assert_eq!('a' as u16, 0x61);
    assert_eq!('a' as u8, 0x61);
    assert_eq!('א' as u8, 0xd0);
    assert_eq!('א' as u16, 0x5d0);
    assert_eq!('א' as u32, 0x5d0);
    assert_eq!('🐵' as u8, 0x35);
    assert_eq!('🐵' as u16, 0xf435);
    assert_eq!('🐵' as u32, 0x1f435);
    assert_eq!('英' as i16, -0x7d0f);
    assert_eq!('英' as u16, 0x82f1);

    // u8-char-cast
    assert_eq!(0x61 as char, 'a');
    assert_eq!(0u8 as char, '\0');
    assert_eq!(0xd7 as char, '×');

    // array-ptr-cast
    let x = [1,2,3];
    let first : *const u32 = &x[0];

    assert_eq!(first, &x as *const _);
    assert_eq!(first, &x as *const u32);

    // fptr-addr-cast
    fn foo() {
        println!("foo!");
    }
    fn bar() {
        println!("bar!");
    }

    assert!(foo as usize != bar as usize);

    // Taking a few bits of a function's address is totally pointless and we detect that
    assert_eq!(foo as i16, foo as usize as i16);

    // fptr-ptr-cast

    assert_eq!(foo as *const u8 as usize, foo as usize);
    assert!(foo as *const u32 != first);
}
fn foo() { }
