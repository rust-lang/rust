//@ edition: 2018

pub fn main() {
    let try_result = try bikeshed Option<_> { //~ ERROR `try bikeshed` expression is experimental
        let x = 5;
        x
    };
    assert_eq!(try_result, Some(5));
}
