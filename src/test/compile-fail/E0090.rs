fn foo<'a: 'b, 'b: 'a>() {}
fn main() {
    foo::<'static>();//~ ERROR E0090
                     //~^ too few lifetime parameters
}
