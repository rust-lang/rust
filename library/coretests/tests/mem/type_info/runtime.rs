use std::any::TypeId;

fn as_unit_ptr<T: 'static>(test_case: &T) -> *const () {
    (test_case as *const T).cast()
}

fn assert_unit_variant_getter<T: 'static>(test_input: &T) {
    let variant_getter = const { TypeId::of::<T>().variant_getter() };
    assert_eq!(variant_getter(as_unit_ptr(test_input)), const { TypeId::of::<T>().variant(0) });
}

#[test]
fn correct_variant_id() {
    use std::any::TypeId;
    #[repr(u128)]
    enum Kitty {
        Sleepy,
        Playful,
        Cuddly = u128::MAX,
    }

    let variant_getter = const { TypeId::of::<Kitty>().variant_getter() };
    assert_eq!(
        variant_getter(as_unit_ptr(&Kitty::Sleepy)),
        const { TypeId::of::<Kitty>().variant(0) }
    );
    assert_eq!(
        variant_getter(as_unit_ptr(&Kitty::Playful)),
        const { TypeId::of::<Kitty>().variant(1) }
    );
    assert_eq!(
        variant_getter(as_unit_ptr(&Kitty::Cuddly)),
        const { TypeId::of::<Kitty>().variant(2) }
    );
}

#[test]
fn variant_getter_for_unit() {
    assert_unit_variant_getter(&());
}

#[test]
fn array() {
    assert_unit_variant_getter::<[u8; 3]>(&[0, 1, 2]);
}

#[test]
fn slice() {
    assert_unit_variant_getter::<&[u8]>(&[0, 1, 2].as_slice());
}

#[test]
fn strukt() {
    assert_unit_variant_getter(&String::new());
}

#[test]
fn union() {
    union TestUnion {
        first: i16,
    }
    let test_union = TestUnion { first: 0 };
    assert_unit_variant_getter(&test_union);
}

#[test]
fn primitives() {
    assert_unit_variant_getter(&true);
    assert_unit_variant_getter(&0u8);
    assert_unit_variant_getter(&'a');
}

#[test]
fn reference_types() {
    assert_unit_variant_getter(&&true);
}

#[test]
fn pointers_types() {
    let test_value = true;
    let test_pointer = &test_value as *const bool;
    assert_unit_variant_getter(&test_pointer);
}

#[test]
fn trait_obj() {
    let test_value = Box::new(String::new());
    let test_object = Box::leak(test_value) as &dyn std::fmt::Debug;
    assert_unit_variant_getter(&test_object);
}

#[test]
fn never_type() {
    let variant_getter = const { TypeId::of::<!>().variant_getter() };
}
