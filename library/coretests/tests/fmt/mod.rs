mod builders;
mod float;
mod num;

#[test]
fn test_format_flags() {
    // No residual flags left by pointer formatting
    let p = "".as_ptr();
    assert_eq!(format!("{:p} {:x}", p, 16), format!("{p:p} 10"));

    assert_eq!(format!("{: >3}", 'a'), "  a");
}

#[test]
fn test_pointer_formats_data_pointer() {
    let b: &[u8] = b"";
    let s: &str = "";
    assert_eq!(format!("{s:p}"), format!("{:p}", s as *const _));
    assert_eq!(format!("{b:p}"), format!("{:p}", b as *const _));
}

#[test]
fn test_fmt_debug_of_raw_pointers() {
    use core::fmt::Debug;
    use core::ptr;

    fn check_fmt<T: Debug>(t: T, start: &str, contains: &str) {
        let formatted = format!("{:?}", t);
        assert!(formatted.starts_with(start), "{formatted:?} doesn't start with {start:?}");
        assert!(formatted.contains(contains), "{formatted:?} doesn't contain {contains:?}");
    }

    assert_eq!(format!("{:?}", ptr::without_provenance_mut::<i32>(0x100)), "0x100");
    assert_eq!(format!("{:?}", ptr::without_provenance::<i32>(0x100)), "0x100");

    let slice = ptr::slice_from_raw_parts(ptr::without_provenance::<i32>(0x100), 3);
    assert_eq!(format!("{:?}", slice as *mut [i32]), "Pointer { addr: 0x100, metadata: 3 }");
    assert_eq!(format!("{:?}", slice as *const [i32]), "Pointer { addr: 0x100, metadata: 3 }");

    let vtable = &mut 500 as &mut dyn Debug;
    check_fmt(vtable as *mut dyn Debug, "Pointer { addr: ", ", metadata: DynMetadata(");
    check_fmt(vtable as *const dyn Debug, "Pointer { addr: ", ", metadata: DynMetadata(");
}

#[test]
fn test_estimated_capacity() {
    assert_eq!(format_args!("").estimated_capacity(), 0);
    assert_eq!(format_args!("{}", { "" }).estimated_capacity(), 0);
    assert_eq!(format_args!("Hello").estimated_capacity(), 5);
    assert_eq!(format_args!("Hello, {}!", { "" }).estimated_capacity(), 16);
    assert_eq!(format_args!("{}, hello!", { "World" }).estimated_capacity(), 0);
    assert_eq!(format_args!("{}. 16-bytes piece", { "World" }).estimated_capacity(), 32);
}

#[test]
fn pad_integral_resets() {
    struct Bar;

    impl core::fmt::Display for Bar {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            "1".fmt(f)?;
            f.pad_integral(true, "", "5")?;
            "1".fmt(f)
        }
    }

    assert_eq!(format!("{Bar:<03}"), "1  0051  ");
}

#[test]
fn test_maybe_uninit_short() {
    // Ensure that the trimmed `MaybeUninit` Debug implementation doesn't break
    let x = core::mem::MaybeUninit::new(0u32);
    assert_eq!(format!("{x:?}"), "MaybeUninit<u32>");
}

#[test]
fn formatting_options_ctor() {
    use core::fmt::FormattingOptions;
    assert_eq!(FormattingOptions::new(), FormattingOptions::default());
}

#[test]
#[allow(deprecated)]
fn formatting_options_flags() {
    use core::fmt::*;
    for sign in [None, Some(Sign::Plus), Some(Sign::Minus)] {
        for alternate in [true, false] {
            for sign_aware_zero_pad in [true, false] {
                for debug_as_hex in [None, Some(DebugAsHex::Lower), Some(DebugAsHex::Upper)] {
                    let mut formatting_options = FormattingOptions::new();
                    formatting_options
                        .sign(sign)
                        .sign_aware_zero_pad(sign_aware_zero_pad)
                        .alternate(alternate)
                        .debug_as_hex(debug_as_hex);

                    assert_eq!(formatting_options.get_sign(), sign);
                    assert_eq!(formatting_options.get_alternate(), alternate);
                    assert_eq!(formatting_options.get_sign_aware_zero_pad(), sign_aware_zero_pad);
                    assert_eq!(formatting_options.get_debug_as_hex(), debug_as_hex);

                    let mut output = String::new();
                    let fmt = Formatter::new(&mut output, formatting_options);
                    assert_eq!(fmt.options(), formatting_options);

                    assert_eq!(fmt.sign_minus(), sign == Some(Sign::Minus));
                    assert_eq!(fmt.sign_plus(), sign == Some(Sign::Plus));
                    assert_eq!(fmt.alternate(), alternate);
                    assert_eq!(fmt.sign_aware_zero_pad(), sign_aware_zero_pad);

                    // The flags method is deprecated.
                    // This checks compatibility with older versions of Rust.
                    assert_eq!(fmt.flags() & 1 != 0, sign == Some(Sign::Plus));
                    assert_eq!(fmt.flags() & 2 != 0, sign == Some(Sign::Minus));
                    assert_eq!(fmt.flags() & 4 != 0, alternate);
                    assert_eq!(fmt.flags() & 8 != 0, sign_aware_zero_pad);
                    assert_eq!(fmt.flags() & 16 != 0, debug_as_hex == Some(DebugAsHex::Lower));
                    assert_eq!(fmt.flags() & 32 != 0, debug_as_hex == Some(DebugAsHex::Upper));
                    assert_eq!(fmt.flags() & 0xFFFF_FFC0, 0);
                }
            }
        }
    }
}
