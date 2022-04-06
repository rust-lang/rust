// Test various uses of structs with distint variances to make sure
// they permit lifetimes to be approximated as expected.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct SomeStruct<T>(fn(T));

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{
    v
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}


fn main() { }
