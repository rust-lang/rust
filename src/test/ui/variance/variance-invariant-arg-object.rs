#![allow(dead_code)]

trait Get<T> : 'static {
    fn get(&self, t: T) -> T;
}

fn get_min_from_max<'min, 'max>(v: Box<Get<&'max i32>>)
                                -> Box<Get<&'min i32>>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn get_max_from_min<'min, 'max, G>(v: Box<Get<&'min i32>>)
                                   -> Box<Get<&'max i32>>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn main() { }
