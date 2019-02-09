fn parse_type(iter: Box<Iterator<Item = &str> + 'static>) -> &str {
//~^ ERROR missing lifetime specifier [E0106]
    iter.next()
}

fn parse_type_2(iter: fn(&u8) -> &u8) -> &str {
//~^ ERROR missing lifetime specifier [E0106]
    iter()
}

fn parse_type_3() -> &str {
//~^ ERROR missing lifetime specifier [E0106]
    unimplemented!()
}

fn main() {}
