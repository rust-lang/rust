// tests/const_conversion_tests.rs
#![feature(const_trait_impl,f16, f128)]
#![feature(ascii_char, ascii_char_variants)]

use std::convert::TryFrom;
use std::net::{Ipv4Addr, IpAddr, SocketAddr};
use std::str::FromStr;
use std::num::NonZero;
use std::ascii::Char as AsciiChar;

const _: () = {
    // --- From<AsciiChar> / From<char> / TryFrom<u32> for char ---
    let ac = AsciiChar::LessThanSign;
    let _: u8   = u8::from(ac);
    let _: u16  = u16::from(ac);
    let _: u32  = u32::from(ac);
    let _: u64  = u64::from(ac);
    let _: u128 = u128::from(ac);
    let _: char = char::from(ac);

    let c = '🚀';
    let _: u32  = u32::from(c);
    let _: u64  = u64::from(c);
    let _: u128 = u128::from(c);
    let _: char = char::from(65u8);
    let _: char = char::try_from(0x1F680u32).unwrap();


    let mut st = String::from("world");
    //~^ ERROR: the trait bound `String: const From<&str>` is not satisfied

    // --- AsRef / AsMut on primitive refs, arrays, strings ---
    let x = &5u8;
    let _: &u8      = x.as_ref();
    //~^ ERROR: the method `as_ref` exists for reference `&u8`, but its trait bounds were not satisfied
    let mut y = 6u8;
    let ym = &mut y;
    let _: &mut u8  = ym.as_mut();
    //~^ ERROR: the method `as_mut` exists for mutable reference `&mut u8`, but its trait bounds were not satisfied

    let arr      = [1u8, 2, 3];
    let _: &[u8] = arr.as_ref();
    //~^ ERROR: the trait bound `[u8; 3]: const AsRef<[u8]>` is not satisfied [E0277]
    let mut arr2 = [4u8, 5, 6];
    let _: &mut [u8] = arr2.as_mut();
    //~^ ERROR: the trait bound `[u8; 3]: const AsMut<[u8]>` is not satisfied [E0277]

    let s  = "hello";
    let _: &str     = s.as_ref();

    // --- bool → integers, isize ↔ usize ---
    let b = true;
    let _: u8    = u8::from(b);
    let _: i8    = i8::from(b);
    let _: u16   = u16::from(b);
    let _: i16   = i16::from(b);
    let _: usize = usize::from(b);
    let _: isize = isize::from(b);

    let _ = usize::try_from(42_isize);
    let _ = isize::try_from(42_usize);

    // --- NonZero conversions ---
    let nz8  = unsafe { NonZero::new_unchecked(5)};
    let _   = NonZero::<u16>::from(nz8);
    let _         = u8::from(nz8);

    // --- IpAddr / SocketAddr ---
    let v4 = Ipv4Addr::new(127, 0, 0, 1);
    let _: u32      = u32::from(v4);
    let _: SocketAddr  = SocketAddr::from((v4, 8080));

    // --- FromStr for ints  ---
    let _: u8   = u8::from_str("123").unwrap();
    let _: i16  = i16::from_str("-456").unwrap();

    ();
};

fn main() {}