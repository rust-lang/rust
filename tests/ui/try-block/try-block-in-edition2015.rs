//@ edition: 2015

pub fn main() {
    let try_result: Option<_> = try {
    //~^ ERROR expected struct, variant or union type, found macro `try`
        let x = 5; //~ ERROR expected identifier, found keyword
        x
    };
    assert_eq!(try_result, Some(5));
}
