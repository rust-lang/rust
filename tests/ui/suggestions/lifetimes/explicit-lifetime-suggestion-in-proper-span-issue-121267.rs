fn main() {}

fn foo(_src: &crate::Foo) -> Option<i32> {
    todo!()
}
fn bar(src: &crate::Foo) -> impl Iterator<Item = i32> {
    [0].into_iter()
 //~^ ERROR hidden type for `impl Iterator<Item = i32>` captures lifetime that does not appear in bounds
        .filter_map(|_| foo(src))
}

struct Foo<'a>(&'a str);
