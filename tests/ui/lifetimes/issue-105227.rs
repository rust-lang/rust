// Regression test for issue #105227.

//@ run-rustfix
#![allow(warnings)]
fn chars0(v :(& str, &str)) -> impl Iterator<Item = char> {
//~^ HELP to declare that `impl Iterator<Item = char>` captures `'_`, you can introduce a named lifetime parameter `'a`
    v.0.chars().chain(v.1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bounds
}

fn chars1(v0 : & str, v1 : &str) -> impl Iterator<Item = char> {
//~^ HELP to declare that `impl Iterator<Item = char>` captures `'_`, you can introduce a named lifetime parameter `'a`
    v0.chars().chain(v1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn chars2<'b>(v0 : &str, v1 : &'_ str, v2 : &'b str) ->
//~^ HELP to declare that `impl Iterator<Item = char>` captures `'_`, you can use the named lifetime parameter `'b`
    (impl Iterator<Item = char>, &'b str)
{
    (v0.chars().chain(v1.chars()), v2)
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn main() {
}
