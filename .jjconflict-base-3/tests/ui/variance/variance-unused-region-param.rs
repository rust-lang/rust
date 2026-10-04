// Test that we report an error for unused type parameters in types.

struct SomeStruct<'a> { x: u32 } //~ ERROR parameter `'a` is never used
enum SomeEnum<'a> { Nothing } //~ ERROR parameter `'a` is never used
union SomeUnion<'a> { x: u32 } //~ ERROR parameter `'a` is never used
trait SomeTrait<'a> { fn foo(&self); } // OK on traits.
type SomeAlias<'a> = u32; // OK on type aliases.

fn main() {}
