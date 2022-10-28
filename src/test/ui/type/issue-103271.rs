fn main() {
    //~^ HELP the following trait is implemented but not in scope; perhaps add a `use` for it:
    let length = <&[_]>::len;
    //~^ ERROR the function or associated item `len` exists for reference `&[_]`, but its trait bounds were not satisfied [E0599]
    //~| function or associated item cannot be called on `&[_]` due to unsatisfied trait bounds
    //~| HELP items from traits can only be used if the trait is in scope
    //~| HELP the function `len` is implemented on `[_]`
    assert_eq!(length(&[1,3]), 2);
}
