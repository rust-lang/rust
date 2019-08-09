// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

struct Struct<T> {
    x: T,
    f: fn(x: T) -> T,
}

fn id<T>(x: T) -> T { x }

impl<T> Struct<T> {

    fn new(x: T) -> Struct<T> {
        Struct {
            x: x,
            f: id
        }
    }

    fn get<T2>(self, x: T2) -> (T, T2) {
        (self.x, x)
    }
}

pub struct LifeTimeOnly<'a> {
    _a: &'a u32
}

impl<'a> LifeTimeOnly<'a> {

    //~ MONO_ITEM fn generic_impl::{{impl}}[1]::foo[0]
    pub fn foo(&self) {}
    //~ MONO_ITEM fn generic_impl::{{impl}}[1]::bar[0]
    pub fn bar(&'a self) {}
    //~ MONO_ITEM fn generic_impl::{{impl}}[1]::baz[0]
    pub fn baz<'b>(&'b self) {}

    pub fn non_instantiated<T>(&self) {}
}

//~ MONO_ITEM fn generic_impl::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::new[0]<i32>
    //~ MONO_ITEM fn generic_impl::id[0]<i32>
    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::get[0]<i32, i16>
    let _ = Struct::new(0i32).get(0i16);

    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::new[0]<i64>
    //~ MONO_ITEM fn generic_impl::id[0]<i64>
    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::get[0]<i64, i16>
    let _ = Struct::new(0i64).get(0i16);

    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::new[0]<char>
    //~ MONO_ITEM fn generic_impl::id[0]<char>
    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::get[0]<char, i16>
    let _ = Struct::new('c').get(0i16);

    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::new[0]<&str>
    //~ MONO_ITEM fn generic_impl::id[0]<&str>
    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::get[0]<generic_impl::Struct[0]<&str>, i16>
    let _ = Struct::new(Struct::new("str")).get(0i16);

    //~ MONO_ITEM fn generic_impl::{{impl}}[0]::new[0]<generic_impl::Struct[0]<&str>>
    //~ MONO_ITEM fn generic_impl::id[0]<generic_impl::Struct[0]<&str>>
    let _ = (Struct::new(Struct::new("str")).f)(Struct::new("str"));

    0
}
