//@ run-pass
#![allow(non_snake_case)]

#[derive(Clone, Copy)]
#[repr(C)]
struct LARGE_INTEGER_U {
    LowPart: u32,
    HighPart: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
union LARGE_INTEGER {
  __unnamed__: LARGE_INTEGER_U,
  u: LARGE_INTEGER_U,
  QuadPart: u64,
}

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    fn increment_all_parts(_: LARGE_INTEGER) -> LARGE_INTEGER;
}

fn main() {
    unsafe {
        let mut li = LARGE_INTEGER { QuadPart: 0 };
        let li_c = increment_all_parts(li);
        li.__unnamed__.LowPart += 1;
        li.__unnamed__.HighPart += 1;
        li.u.LowPart += 1;
        li.u.HighPart += 1;
        li.QuadPart += 1;
        assert_eq!(li.QuadPart, li_c.QuadPart);
    }
}
