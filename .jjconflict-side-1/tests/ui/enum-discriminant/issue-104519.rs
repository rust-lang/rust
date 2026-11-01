//@ run-pass
#![allow(dead_code)]

enum OpenResult {
    Ok(()),
    Err(()),
    TransportErr(TransportErr),
}

#[repr(i32)]
enum TransportErr {
    UnknownMethod = -2,
}

#[inline(never)]
fn some_match(result: OpenResult) -> u8 {
    match result {
        OpenResult::Ok(()) => 0,
        _ => 1,
    }
}

fn main() {
    let result = OpenResult::Ok(());
    assert_eq!(some_match(result), 0);
}
