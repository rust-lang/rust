// compile-flags:-Zprint-mono-items=eager

#![deny(dead_code)]
#![feature(start)]

pub trait SomeTrait {
    fn foo(&self);
    fn bar<T>(&self, x: T);
}

impl SomeTrait for i64 {

    //~ MONO_ITEM fn trait_implementations::{{impl}}[0]::foo[0]
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

impl SomeTrait for i32 {

    //~ MONO_ITEM fn trait_implementations::{{impl}}[1]::foo[0]
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

pub trait SomeGenericTrait<T> {
    fn foo(&self, x: T);
    fn bar<T2>(&self, x: T, y: T2);
}

// Concrete impl of generic trait
impl SomeGenericTrait<u32> for f64 {

    //~ MONO_ITEM fn trait_implementations::{{impl}}[2]::foo[0]
    fn foo(&self, _: u32) {}

    fn bar<T2>(&self, _: u32, _: T2) {}
}

// Generic impl of generic trait
impl<T> SomeGenericTrait<T> for f32 {

    fn foo(&self, _: T) {}
    fn bar<T2>(&self, _: T, _: T2) {}
}

//~ MONO_ITEM fn trait_implementations::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
   //~ MONO_ITEM fn trait_implementations::{{impl}}[1]::bar[0]<char>
   0i32.bar('x');

   //~ MONO_ITEM fn trait_implementations::{{impl}}[2]::bar[0]<&str>
   0f64.bar(0u32, "&str");

   //~ MONO_ITEM fn trait_implementations::{{impl}}[2]::bar[0]<()>
   0f64.bar(0u32, ());

   //~ MONO_ITEM fn trait_implementations::{{impl}}[3]::foo[0]<char>
   0f32.foo('x');

   //~ MONO_ITEM fn trait_implementations::{{impl}}[3]::foo[0]<i64>
   0f32.foo(-1i64);

   //~ MONO_ITEM fn trait_implementations::{{impl}}[3]::bar[0]<u32, ()>
   0f32.bar(0u32, ());

   //~ MONO_ITEM fn trait_implementations::{{impl}}[3]::bar[0]<&str, &str>
   0f32.bar("&str", "&str");

   0
}
