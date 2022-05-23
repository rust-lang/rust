// compile-flags: -Zmir-opt-level=4

pub fn bar<T>(s: &'static mut ())
where
    &'static mut (): Clone, //~ ERROR the trait bound
{
    <&'static mut () as Clone>::clone(&s);
}

fn main() {}
