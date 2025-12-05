//! The drop check is currently more permissive when match arms have guards, due to eagerly creating
//! by-ref bindings for the guard (#142057).

struct Struct<T>(T);
impl<T> Drop for Struct<T> {
    fn drop(&mut self) {}
}

fn main() {
    // This is an error: `short1` is dead before `long1` is dropped.
    match (Struct(&&0), 1) {
        (mut long1, ref short1) => long1.0 = &short1,
        //~^ ERROR `short1` does not live long enough
    }
    // This is OK: `short2`'s storage is live until after `long2`'s drop runs.
    match (Struct(&&0), 1) {
        (mut long2, ref short2) if true => long2.0 = &short2,
        _ => unreachable!(),
    }
    // This depends on the binding modes of the first or-pattern alternatives:
    let res: &Result<u8, &u8> = &Ok(1);
    match (Struct(&&0), res) {
        (mut long3, Ok(short3) | &Err(short3)) if true => long3.0 = &short3,
        _ => unreachable!(),
    }
    match (Struct(&&0), res) {
        (mut long4, &Err(short4) | Ok(short4)) if true => long4.0 = &short4,
        //~^ ERROR `short4` does not live long enough
        _ => unreachable!(),
    }
}
