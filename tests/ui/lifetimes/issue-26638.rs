fn parse_type(iter: Box<dyn Iterator<Item=&str>+'static>) -> &str { iter.next() }
//~^ ERROR missing lifetime specifier [E0106]

fn parse_type_2(iter: fn(&u8)->&u8) -> &str { iter() }
//~^ ERROR missing lifetime specifier [E0106]
//~| ERROR function takes 1 argument but 0 arguments were supplied

fn parse_type_3() -> &str { unimplemented!() }
//~^ ERROR missing lifetime specifier [E0106]

fn main() {}
