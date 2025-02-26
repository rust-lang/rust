use core::any::*;

#[derive(PartialEq, Debug)]
struct Test;

static TEST: &'static str = "Test";

#[test]
fn any_referenced() {
    let (a, b, c) = (&5 as &dyn Any, &TEST as &dyn Any, &Test as &dyn Any);

    assert!(a.is::<i32>());
    assert!(!b.is::<i32>());
    assert!(!c.is::<i32>());

    assert!(!a.is::<&'static str>());
    assert!(b.is::<&'static str>());
    assert!(!c.is::<&'static str>());

    assert!(!a.is::<Test>());
    assert!(!b.is::<Test>());
    assert!(c.is::<Test>());
}

#[test]
fn any_owning() {
    let (a, b, c) = (
        Box::new(5_usize) as Box<dyn Any>,
        Box::new(TEST) as Box<dyn Any>,
        Box::new(Test) as Box<dyn Any>,
    );

    assert!(a.is::<usize>());
    assert!(!b.is::<usize>());
    assert!(!c.is::<usize>());

    assert!(!a.is::<&'static str>());
    assert!(b.is::<&'static str>());
    assert!(!c.is::<&'static str>());

    assert!(!a.is::<Test>());
    assert!(!b.is::<Test>());
    assert!(c.is::<Test>());
}

#[test]
fn any_downcast_ref() {
    let a = &5_usize as &dyn Any;

    match a.downcast_ref::<usize>() {
        Some(&5) => {}
        x => panic!("Unexpected value {x:?}"),
    }

    match a.downcast_ref::<Test>() {
        None => {}
        x => panic!("Unexpected value {x:?}"),
    }
}

#[test]
fn any_downcast_mut() {
    let mut a = 5_usize;
    let mut b: Box<_> = Box::new(7_usize);

    let a_r = &mut a as &mut dyn Any;
    let tmp: &mut usize = &mut *b;
    let b_r = tmp as &mut dyn Any;

    match a_r.downcast_mut::<usize>() {
        Some(x) => {
            assert_eq!(*x, 5);
            *x = 612;
        }
        x => panic!("Unexpected value {x:?}"),
    }

    match b_r.downcast_mut::<usize>() {
        Some(x) => {
            assert_eq!(*x, 7);
            *x = 413;
        }
        x => panic!("Unexpected value {x:?}"),
    }

    match a_r.downcast_mut::<Test>() {
        None => (),
        x => panic!("Unexpected value {x:?}"),
    }

    match b_r.downcast_mut::<Test>() {
        None => (),
        x => panic!("Unexpected value {x:?}"),
    }

    match a_r.downcast_mut::<usize>() {
        Some(&mut 612) => {}
        x => panic!("Unexpected value {x:?}"),
    }

    match b_r.downcast_mut::<usize>() {
        Some(&mut 413) => {}
        x => panic!("Unexpected value {x:?}"),
    }
}

#[test]
fn any_fixed_vec() {
    let test = [0_usize; 8];
    let test = &test as &dyn Any;
    assert!(test.is::<[usize; 8]>());
    assert!(!test.is::<[usize; 10]>());
}

#[test]
fn any_unsized() {
    fn is_any<T: Any + ?Sized>() {}
    is_any::<[i32]>();
}

#[cfg(feature = "debug_typeid")]
#[test]
fn debug_typeid_includes_name() {
    let type_id = TypeId::of::<[usize; 2]>();
    let debug_str = format!("{type_id:?}");
    assert!(debug_str.ends_with("= [usize; 2])"), "{debug_str:?} did not match");
}

#[test]
fn distinct_type_names() {
    // https://github.com/rust-lang/rust/issues/84666

    struct Velocity(#[allow(dead_code)] f32, #[allow(dead_code)] f32);

    fn type_name_of_val<T>(_: T) -> &'static str {
        type_name::<T>()
    }

    assert_ne!(type_name_of_val(Velocity), type_name_of_val(Velocity(0.0, -9.8)),);
}

#[test]
fn dyn_type_name() {
    trait Foo {
        type Bar;
    }

    assert_eq!(
        "dyn core::ops::function::Fn(i32, i32) -> i32",
        std::any::type_name::<dyn Fn(i32, i32) -> i32>()
    );
    assert_eq!(
        "dyn coretests::any::dyn_type_name::Foo<Bar = i32> \
        + core::marker::Send + core::marker::Sync",
        std::any::type_name::<dyn Foo<Bar = i32> + Send + Sync>()
    );
}
