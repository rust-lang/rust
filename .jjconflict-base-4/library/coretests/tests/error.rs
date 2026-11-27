use core::error::{Request, request_ref, request_value};

// Test the `Request` API.
#[derive(Debug)]
struct SomeConcreteType {
    some_string: String,
}

impl std::fmt::Display for SomeConcreteType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A")
    }
}

impl std::error::Error for SomeConcreteType {
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        request
            .provide_ref::<String>(&self.some_string)
            .provide_ref::<str>(&self.some_string)
            .provide_value_with::<String>(|| "bye".to_owned());
    }
}

// Test the Error.provide and request mechanisms with a by-reference trait object.
#[test]
fn test_error_generic_member_access() {
    let obj = &SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
}

// Test the Error.provide and request mechanisms with a by-reference trait object.
#[test]
fn test_request_constructor() {
    let obj: &dyn std::error::Error = &SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
}

// Test the Error.provide and request mechanisms with a boxed trait object.
#[test]
fn test_error_generic_member_access_boxed() {
    let obj: Box<dyn std::error::Error> =
        Box::new(SomeConcreteType { some_string: "hello".to_owned() });

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");

    // NOTE: Box<E> only implements Error when E: Error + Sized, which means we can't pass a
    // Box<dyn Error> to request_value.
    //assert_eq!(request_value::<String>(&obj).unwrap(), "bye");
}

// Test the Error.provide and request mechanisms with a concrete object.
#[test]
fn test_error_generic_member_access_concrete() {
    let obj = SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(request_ref::<String>(&obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
}
