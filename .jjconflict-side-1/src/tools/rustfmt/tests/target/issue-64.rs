// Regression test for issue 64

pub fn header_name<T: Header>() -> &'static str {
    let name = <T as Header>::header_name();
    let func = <T as Header>::header_name;
    name
}
