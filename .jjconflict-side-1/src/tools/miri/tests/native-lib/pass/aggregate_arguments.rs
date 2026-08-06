fn main() {
    test_pass_struct();
    test_pass_struct_complex();
    test_return_struct();
}

/// Test passing a basic struct as an argument.
fn test_pass_struct() {
    // Exactly two fields, so that we hit the ScalarPair case.
    #[repr(C)]
    struct PassMe {
        value: i32,
        other_value: i64,
    }

    extern "C" {
        fn pass_struct(s: PassMe) -> i64;
    }

    let pass_me = PassMe { value: 42, other_value: 1337 };
    assert_eq!(unsafe { pass_struct(pass_me) }, 42 + 1337);
}

fn test_return_struct() {
    // Exactly two fields, so that we hit the ScalarPair case.
    #[repr(C)]
    struct PassMe {
        value: i32,
        other_value: i64,
    }

    extern "C" {
        fn return_struct(v: i32, ov: i64) -> PassMe;
    }

    let pass_me = unsafe { return_struct(1, 2) };
    assert_eq!(pass_me.value, 1);
    assert_eq!(pass_me.other_value, 2);
}

/// Test passing a more complex struct as an argument.
fn test_pass_struct_complex() {
    #[repr(C)]
    struct ComplexStruct {
        part_1: Part1,
        part_2: Part2,
        part_3: u32,
    }
    #[repr(C)]
    struct Part1 {
        high: u16,
        low: u16,
    }
    #[repr(C)]
    struct Part2 {
        bits: u32,
    }

    extern "C" {
        fn pass_struct_complex(s: ComplexStruct, high: u16, low: u16, bits: u32) -> i32;
    }

    let high = 0xabcd;
    let low = 0xef01;
    let bits = 0xabcdef01;

    let complex =
        ComplexStruct { part_1: Part1 { high, low }, part_2: Part2 { bits }, part_3: bits };
    assert_eq!(unsafe { pass_struct_complex(complex, high, low, bits) }, 0);
}
