// run-pass
pub trait Test { type T; }

impl Test for u32 {
    type T = i32;
}

pub mod export {
    #[no_mangle]
    pub extern "C" fn issue_28983(t: <u32 as ::Test>::T) -> i32 { t*3 }
}

// to test both exporting and importing functions, import
// a function from ourselves.
extern "C" {
    fn issue_28983(t: <u32 as Test>::T) -> i32;
}

fn main() {
    assert_eq!(export::issue_28983(2), 6);
    assert_eq!(unsafe { issue_28983(3) }, 9);
}
