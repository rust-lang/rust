// Test that a covariant struct does not permit the lifetime of a
// reference to be enlarged.

struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn main() { }
