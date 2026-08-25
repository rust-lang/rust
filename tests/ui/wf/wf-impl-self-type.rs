// Tests that we point at the proper location for an error
// involving the self-type of an impl

trait Foo {}
impl Foo for Option<[u8]> {} //~ ERROR the size for

fn main() {}
