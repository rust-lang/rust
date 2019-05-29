#![allow(dead_code)]

trait Get<T> : 'static {
    fn get(&self, t: T) -> T;
}

fn get_min_from_max<'min, 'max>(v: Box<dyn Get<&'max i32>>)
                                -> Box<dyn Get<&'min i32>>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn get_max_from_min<'min, 'max, G>(v: Box<dyn Get<&'min i32>>)
                                   -> Box<dyn Get<&'max i32>>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn main() { }
