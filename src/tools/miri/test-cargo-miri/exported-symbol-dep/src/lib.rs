#[no_mangle]
fn exported_symbol() -> i32 {
    123456
}

struct AssocFn;

impl AssocFn {
    #[no_mangle]
    fn assoc_fn_as_exported_symbol() -> i32 {
        -123456
    }
}
