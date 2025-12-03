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

struct Invalid;
#[derive(Debug, PartialEq, Eq)]
struct Valid;

impl std::error::Error for SomeConcreteType {
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        request
            .provide_ref::<String>(&self.some_string)
            .provide_ref::<str>(&self.some_string)
            .provide_value_with::<String>(|| "bye".to_owned())
            .provide_value_with::<Invalid>(|| panic!("should not be called"));
        if request.would_be_satisfied_by_ref_of::<Invalid>() {
            panic!("should not be satisfied");
        }
        if request.would_be_satisfied_by_ref_of::<Valid>() {
            request.provide_ref(&Valid);
        }
    }
}

// Test the Error.provide and request mechanisms with a by-reference trait object.
#[test]
fn test_error_generic_member_access() {
    let obj = &SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
    assert_eq!(request_ref::<Valid>(&obj), Some(&Valid));
}

// Test the Error.provide and request mechanisms with a by-reference trait object.
#[test]
fn test_request_constructor() {
    let obj: &dyn std::error::Error = &SomeConcreteType { some_string: "hello".to_owned() };

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_value::<u8>(&obj), None);
    assert_eq!(request_ref::<Valid>(&obj), Some(&Valid));
}

// Test the Error.provide and request mechanisms with a boxed trait object.
#[test]
fn test_error_generic_member_access_boxed() {
    let obj: Box<dyn std::error::Error> =
        Box::new(SomeConcreteType { some_string: "hello".to_owned() });

    assert_eq!(request_ref::<String>(&*obj).unwrap(), "hello");
    assert_eq!(request_value::<String>(&*obj).unwrap(), "bye");
    assert_eq!(request_ref::<Valid>(&*obj), Some(&Valid));

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
    assert_eq!(request_ref::<Valid>(&obj), Some(&Valid));
}

#[test]
fn test_error_combined_access_concrete() {
    let obj = SomeConcreteType { some_string: "hello".to_owned() };

    let mut string_val = None;
    let mut string_ref = None;
    let mut u8_val = None;
    let mut valid_ref = None;

    MultiRequestBuilder::new()
        .with_value::<String>()
        .with_ref::<String>()
        .with_value::<u8>()
        .with_ref::<Valid>()
        .request(&obj)
        .retrieve_value::<String>(|val| string_val = Some(val))
        .retrieve_ref::<String>(|val| string_ref = Some(val))
        .retrieve_value::<u8>(|val| u8_val = Some(val))
        .retrieve_ref::<Valid>(|val| valid_ref = Some(val));

    assert_eq!(string_ref.unwrap(), "hello");
    assert_eq!(string_val.unwrap(), "bye");
    assert_eq!(u8_val, None);
    assert_eq!(valid_ref.unwrap(), Valid);
}

#[test]
fn test_error_combined_access_dyn() {
    let obj = SomeConcreteType { some_string: "hello".to_owned() };
    let obj: &dyn Error = &obj;

    let mut string_val = None;
    let mut string_ref = None;
    let mut u8_val = None;
    let mut valid_ref = None;

    MultiRequestBuilder::new()
        .with_value::<String>()
        .with_ref::<String>()
        .with_value::<u8>()
        .with_ref::<Valid>()
        .request(&obj)
        .retrieve_value::<String>(|val| string_val = Some(val))
        .retrieve_ref::<String>(|val| string_ref = Some(val))
        .retrieve_value::<u8>(|val| u8_val = Some(val))
        .retrieve_ref::<Valid>(|val| valid_ref = Some(val));

    assert_eq!(string_ref.unwrap(), "hello");
    assert_eq!(string_val.unwrap(), "bye");
    assert_eq!(u8_val, None);
    assert_eq!(valid_ref.unwrap(), Valid);
}
