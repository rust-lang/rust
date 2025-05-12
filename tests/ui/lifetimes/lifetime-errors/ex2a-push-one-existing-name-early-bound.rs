trait Foo<'a> {}
impl<'a, T> Foo<'a> for T {}

fn baz<'a, 'b, T>(x: &mut Vec<&'a T>, y: &T)
    where i32: Foo<'a>,
          u32: Foo<'b>
{
    x.push(y); //~ ERROR explicit lifetime required
}
fn main() {
}
