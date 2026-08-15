// https://github.com/rust-lang/rust/issues/30236
type Foo<
    Unused //~ ERROR type parameter `Unused` is never used
    > = u8;

fn main() {

}
