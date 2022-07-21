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

#[test]
fn distinct_type_names() {
    // https://github.com/rust-lang/rust/issues/84666

    struct Velocity(f32, f32);

    fn type_name_of_val<T>(_: T) -> &'static str {
        type_name::<T>()
    }

    assert_ne!(type_name_of_val(Velocity), type_name_of_val(Velocity(0.0, -9.8)),);
}

// Test the `Provider` API.

struct SomeConcreteType {
    some_string: String,
}

impl Provider for SomeConcreteType {
    fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
        demand
            .provide_ref::<String>(&self.some_string)
            .provide_ref::<str>(&self.some_string)
            .provide_value_with::<String>(|| "bye".to_owned());
    }
}

// Test the provide and request mechanisms with a by-reference trait object.
#[test]
fn test_provider() {
    let obj: &dyn Provider = &SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(&**request_ref::<String>(obj).unwrap(), "hello");
    assert_eq!(&*request_value::<String>(obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(obj), None);
}

// Test the provide and request mechanisms with a boxed trait object.
#[test]
fn test_provider_boxed() {
    let obj: Box<dyn Provider> = Box::new(SomeConcreteType { some_string: "hello".to_owned() });

    assert_eq!(&**request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(&*request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&*obj), None);
}

// Test the provide and request mechanisms with a concrete object.
#[test]
fn test_provider_concrete() {
    let obj = SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(&**request_ref::<String>(&obj).unwrap(), "hello");
    assert_eq!(&*request_value::<String>(&obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
}

trait OtherTrait: Provider {}

impl OtherTrait for SomeConcreteType {}

impl dyn OtherTrait {
    fn get_ref<T: 'static + ?Sized>(&self) -> Option<&T> {
        request_ref::<T>(self)
    }
}

// Test the provide and request mechanisms via an intermediate trait.
#[test]
fn test_provider_intermediate() {
    let obj: &dyn OtherTrait = &SomeConcreteType { some_string: "hello".to_owned() };
    assert_eq!(obj.get_ref::<str>().unwrap(), "hello");
}
