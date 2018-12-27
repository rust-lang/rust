// Test various uses of structs with distint variances to make sure
// they permit lifetimes to be approximated as expected.



struct SomeStruct<T>(*mut T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{
    v //~ ERROR mismatched types
}


fn main() { }
