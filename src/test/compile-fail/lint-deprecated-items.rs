#[forbid(deprecated_item)];

type Bar = uint;

#[deprecated = "use Bar instead"]
type Foo = int;

fn main() {
    let _x: Foo = 21;
}
