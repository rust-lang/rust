fn foo<T>(t: T) -> i32
    where T : for<'a> Fn(&'a u8) -> i32,
          T : for<'b> Fn(&'b u8) -> i32,
{
    t(&3)
}

fn main() {
}
