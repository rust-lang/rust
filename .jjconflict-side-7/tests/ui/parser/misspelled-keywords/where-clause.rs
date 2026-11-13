fn code<T>() -> u8
wheree
//~^ ERROR expected one of
    T: Debug,
{
}

fn main() {}
