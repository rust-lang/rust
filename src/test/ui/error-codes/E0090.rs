fn foo<'a: 'b, 'b: 'a>() {}

fn main() {
    foo::<'static>(); //~ ERROR wrong number of lifetime arguments: expected 2, found 1 [E0090]
}
