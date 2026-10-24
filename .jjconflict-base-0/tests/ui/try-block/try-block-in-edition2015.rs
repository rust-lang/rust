//@ edition: 2015

pub fn main() {
    let try_result: Option<_> = try {
    //~^ ERROR cannot find struct, variant or union type `try` in this scope
        let x = 5; //~ ERROR expected identifier, found keyword
        x
    };
    assert_eq!(try_result, Some(5));
}
