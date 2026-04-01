//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

pub trait SomeTrait {
    fn foo(&self);
    fn bar<T>(&self, x: T);
}

impl SomeTrait for i64 {
    //~ MONO_ITEM fn <i64 as SomeTrait>::foo
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

impl SomeTrait for i32 {
    //~ MONO_ITEM fn <i32 as SomeTrait>::foo
    fn foo(&self) {}

    fn bar<T>(&self, _: T) {}
}

pub trait SomeGenericTrait<T> {
    fn foo(&self, x: T);
    fn bar<T2>(&self, x: T, y: T2);
}

// Concrete impl of generic trait
impl SomeGenericTrait<u32> for f64 {
    //~ MONO_ITEM fn <f64 as SomeGenericTrait<u32>>::foo
    fn foo(&self, _: u32) {}

    fn bar<T2>(&self, _: u32, _: T2) {}
}

// Generic impl of generic trait
impl<T> SomeGenericTrait<T> for f32 {
    fn foo(&self, _: T) {}
    fn bar<T2>(&self, _: T, _: T2) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn <i32 as SomeTrait>::bar::<char>
    0i32.bar('x');

    //~ MONO_ITEM fn <f64 as SomeGenericTrait<u32>>::bar::<&str>
    0f64.bar(0u32, "&str");

    //~ MONO_ITEM fn <f64 as SomeGenericTrait<u32>>::bar::<()>
    0f64.bar(0u32, ());

    //~ MONO_ITEM fn <f32 as SomeGenericTrait<char>>::foo
    0f32.foo('x');

    //~ MONO_ITEM fn <f32 as SomeGenericTrait<i64>>::foo
    0f32.foo(-1i64);

    //~ MONO_ITEM fn <f32 as SomeGenericTrait<u32>>::bar::<()>
    0f32.bar(0u32, ());

    //~ MONO_ITEM fn <f32 as SomeGenericTrait<&str>>::bar::<&str>
    0f32.bar("&str", "&str");

    0
}
