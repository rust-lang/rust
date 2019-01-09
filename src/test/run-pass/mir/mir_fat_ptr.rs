// run-pass
// test that ordinary fat pointer operations work.

struct Wrapper<T: ?Sized>(u32, T);

struct FatPtrContainer<'a> {
    ptr: &'a [u8]
}

fn fat_ptr_project(a: &Wrapper<[u8]>) -> &[u8] {
    &a.1
}

fn fat_ptr_simple(a: &[u8]) -> &[u8] {
    a
}

fn fat_ptr_via_local(a: &[u8]) -> &[u8] {
    let x = a;
    x
}

fn fat_ptr_from_struct(s: FatPtrContainer) -> &[u8] {
    s.ptr
}

fn fat_ptr_to_struct(a: &[u8]) -> FatPtrContainer {
    FatPtrContainer { ptr: a }
}

fn fat_ptr_store_to<'a>(a: &'a [u8], b: &mut &'a [u8]) {
    *b = a;
}

fn fat_ptr_constant() -> &'static str {
    "HELLO"
}

fn main() {
    let a = Wrapper(4, [7,6,5]);

    let p = fat_ptr_project(&a);
    let p = fat_ptr_simple(p);
    let p = fat_ptr_via_local(p);
    let p = fat_ptr_from_struct(fat_ptr_to_struct(p));

    let mut target : &[u8] = &[42];
    fat_ptr_store_to(p, &mut target);
    assert_eq!(target, &a.1);

    assert_eq!(fat_ptr_constant(), "HELLO");
}
