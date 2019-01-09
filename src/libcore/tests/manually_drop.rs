use core::mem::ManuallyDrop;

#[test]
fn smoke() {
    struct TypeWithDrop;
    impl Drop for TypeWithDrop {
        fn drop(&mut self) {
            unreachable!("Should not get dropped");
        }
    }

    let x = ManuallyDrop::new(TypeWithDrop);
    drop(x);

    // also test unsizing
    let x : Box<ManuallyDrop<[TypeWithDrop]>> =
        Box::new(ManuallyDrop::new([TypeWithDrop, TypeWithDrop]));
    drop(x);
}
