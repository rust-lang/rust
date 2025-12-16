use core::error::provide::{MultiRequestBuilder, MultiResponse, Request};
use core::error::{Error, request_ref, request_value};

struct Invalid;
#[derive(Debug, PartialEq, Eq)]
struct Valid;

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

    let mut request = MultiRequestBuilder::new()
        .with_value::<String>()
        .with_ref::<String>()
        .with_value::<u8>()
        .with_ref::<Valid>()
        .request(&obj);

    assert_eq!(request.retrieve_ref::<String>().unwrap(), "hello");
    assert_eq!(request.retrieve_value::<String>().unwrap(), "bye");
    assert_eq!(request.retrieve_value::<u8>(), None);
    assert_eq!(request.retrieve_ref::<Valid>().unwrap(), &Valid);

    // second retrieve of same value returns none
    assert_eq!(request.retrieve_ref::<Valid>(), None);
    assert_eq!(request.retrieve_value::<String>(), None);
    // retrieving an unknown type should return none
    assert_eq!(request.retrieve_value::<*const ()>(), None);
}

#[test]
fn test_error_combined_access_dyn() {
    let obj = SomeConcreteType { some_string: "hello".to_owned() };
    let obj: &dyn Error = &obj;

    let mut request = MultiRequestBuilder::new()
        .with_value::<String>()
        .with_ref::<String>()
        .with_value::<u8>()
        .with_ref::<Valid>()
        .request(&obj);

    assert_eq!(request.retrieve_ref::<String>().unwrap(), "hello");
    assert_eq!(request.retrieve_value::<String>().unwrap(), "bye");
    assert_eq!(request.retrieve_value::<u8>(), None);
    assert_eq!(request.retrieve_ref::<Valid>().unwrap(), &Valid);

    // second retrieve of same value returns none
    assert_eq!(request.retrieve_ref::<Valid>(), None);
    assert_eq!(request.retrieve_value::<String>(), None);
    // retrieving an unknown type should return none
    assert_eq!(request.retrieve_value::<*const ()>(), None);
}

#[derive(Debug)]
struct ProvideMultipleTimes;

impl std::fmt::Display for ProvideMultipleTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A")
    }
}

impl std::error::Error for ProvideMultipleTimes {
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        let previous_satisfied_by_ref_of_string = request.would_be_satisfied_by_ref_of::<String>();
        request.provide_value::<String>("good".to_string());
        assert!(
            !request.would_be_satisfied_by_value_of::<String>(),
            "already provided String by value"
        );
        // "overwriting" the String should just be ignored
        request.provide_value::<String>("bad".to_string());
        // but providing a ref works
        if previous_satisfied_by_ref_of_string {
            assert!(request.would_be_satisfied_by_ref_of::<String>());
            request.provide_ref::<String>(const { &String::new() });
        }
    }
}

#[test]
fn test_provide_multiple_times_single() {
    let obj = ProvideMultipleTimes;
    let obj: &dyn Error = &obj;

    assert_eq!(*request_ref::<String>(&*obj).unwrap(), String::new());
    assert_eq!(request_value::<String>(&*obj).unwrap(), "good");
}

#[test]
fn test_provide_multiple_times_multi() {
    let obj = ProvideMultipleTimes;
    let obj: &dyn Error = &obj;

    let mut request =
        MultiRequestBuilder::new().with_value::<String>().with_ref::<String>().request(obj);

    assert_eq!(*request.retrieve_ref::<String>().unwrap(), String::new());
    assert_eq!(request.retrieve_value::<String>().unwrap(), "good");
}

fn assert_copy<T: Copy>(_t: T) {}

#[test]
fn test_builder_copy_and_debug() {
    // Check that MultiRequestBuilder implements Debug + Copy even if the contents doesn't (the exact contents don't matter)
    // (Chain*MultiRequest don't, but their values are not really user-visible so it doesn't matter)
    let builder = MultiRequestBuilder::new().with_value::<Valid>().with_ref::<Invalid>();
    assert_copy(builder);
    // check Debug
    assert_eq!(
        format!("{:?}", builder),
        "MultiRequestBuilder(\"core::error::provide::ChainRefMultiRequestBuilder<coretests::error::Invalid, core::error::provide::ChainValMultiRequestBuilder<coretests::error::Valid, core::error::provide::EmptyMultiRequestBuilder>>\")"
    );
}
