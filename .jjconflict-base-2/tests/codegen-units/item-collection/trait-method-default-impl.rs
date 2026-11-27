//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

trait SomeTrait {
    fn foo(&self) {}
    fn bar<T>(&self, x: T) -> T {
        x
    }
}

impl SomeTrait for i8 {
    // take the default implementations

    // For the non-generic foo(), we should generate a codegen-item even if it
    // is not called anywhere
    //~ MONO_ITEM fn <i8 as SomeTrait>::foo
}

trait SomeGenericTrait<T1> {
    fn foo(&self) {}
    fn bar<T2>(&self, x: T1, y: T2) {}
}

// Non-generic impl of generic trait
impl SomeGenericTrait<u64> for i32 {
    // take the default implementations

    // For the non-generic foo(), we should generate a codegen-item even if it
    // is not called anywhere
    //~ MONO_ITEM fn <i32 as SomeGenericTrait<u64>>::foo
}

// Non-generic impl of generic trait
impl<T1> SomeGenericTrait<T1> for u32 {
    // take the default implementations
    // since nothing is monomorphic here, nothing should be generated unless used somewhere.
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn <i8 as SomeTrait>::bar::<char>
    let _ = 1i8.bar('c');

    //~ MONO_ITEM fn <i8 as SomeTrait>::bar::<&str>
    let _ = 2i8.bar("&str");

    //~ MONO_ITEM fn <i32 as SomeGenericTrait<u64>>::bar::<char>
    0i32.bar(0u64, 'c');

    //~ MONO_ITEM fn <i32 as SomeGenericTrait<u64>>::bar::<&str>
    0i32.bar(0u64, "&str");

    //~ MONO_ITEM fn <u32 as SomeGenericTrait<i8>>::bar::<&[char; 1]>
    0u32.bar(0i8, &['c']);

    //~ MONO_ITEM fn <u32 as SomeGenericTrait<i16>>::bar::<()>
    0u32.bar(0i16, ());

    0i8.foo();
    0i32.foo();

    0
}
