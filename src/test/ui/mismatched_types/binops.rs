fn main() {
    1 + Some(1); //~ ERROR cannot add `std::option::Option<{integer}>` to `{integer}`
    2 as usize - Some(1); //~ ERROR cannot subtract `std::option::Option<{integer}>` from `usize`
    3 * (); //~ ERROR cannot multiply `()` to `{integer}`
    4 / ""; //~ ERROR cannot divide `{integer}` by `&str`
    5 < String::new(); //~ ERROR can't compare `{integer}` with `std::string::String`
    6 == Ok(1); //~ ERROR can't compare `{integer}` with `std::result::Result<{integer}, _>`
}
