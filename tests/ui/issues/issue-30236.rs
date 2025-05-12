type Foo<
    Unused //~ ERROR type parameter `Unused` is never used
    > = u8;

fn main() {

}
