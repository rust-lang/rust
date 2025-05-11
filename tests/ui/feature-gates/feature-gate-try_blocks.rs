//@ edition: 2018

pub fn main() {
    let try_result: Option<_> = try { //~ ERROR `try` expression is experimental
        let x = 5;
        x
    };
    assert_eq!(try_result, Some(5));
}
