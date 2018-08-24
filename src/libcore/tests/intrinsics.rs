use core::any::TypeId;

#[test]
fn test_typeid_sized_types() {
    struct X; struct Y(u32);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}

#[test]
fn test_typeid_unsized_types() {
    trait Z {}
    struct X(str); struct Y(dyn Z + 'static);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}
