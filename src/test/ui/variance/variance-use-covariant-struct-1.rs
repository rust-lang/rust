// Test that a covariant struct does not permit the lifetime of a
// reference to be enlarged.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
