//@ compile-flags:-Clink-dead-code -Zinline-mir=no

#![deny(dead_code)]
#![crate_type = "lib"]

struct Struct<T> {
    x: T,
    f: fn(x: T) -> T,
}

fn id<T>(x: T) -> T {
    x
}

impl<T> Struct<T> {
    fn new(x: T) -> Struct<T> {
        Struct { x, f: id }
    }

    fn get<T2>(self, x: T2) -> (T, T2) {
        (self.x, x)
    }
}

pub struct LifeTimeOnly<'a> {
    _a: &'a u32,
}

impl<'a> LifeTimeOnly<'a> {
    //~ MONO_ITEM fn LifeTimeOnly::<'_>::foo
    pub fn foo(&self) {}
    //~ MONO_ITEM fn LifeTimeOnly::<'_>::bar
    pub fn bar(&'a self) {}
    //~ MONO_ITEM fn LifeTimeOnly::<'_>::baz
    pub fn baz<'b>(&'b self) {}

    pub fn non_instantiated<T>(&self) {}
}

//~ MONO_ITEM fn start
#[no_mangle]
pub fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn Struct::<i32>::new
    //~ MONO_ITEM fn id::<i32>
    //~ MONO_ITEM fn Struct::<i32>::get::<i16>
    let _ = Struct::new(0i32).get(0i16);

    //~ MONO_ITEM fn Struct::<i64>::new
    //~ MONO_ITEM fn id::<i64>
    //~ MONO_ITEM fn Struct::<i64>::get::<i16>
    let _ = Struct::new(0i64).get(0i16);

    //~ MONO_ITEM fn Struct::<char>::new
    //~ MONO_ITEM fn id::<char>
    //~ MONO_ITEM fn Struct::<char>::get::<i16>
    let _ = Struct::new('c').get(0i16);

    //~ MONO_ITEM fn Struct::<&str>::new
    //~ MONO_ITEM fn id::<&str>
    //~ MONO_ITEM fn Struct::<Struct<&str>>::get::<i16>
    let _ = Struct::new(Struct::new("str")).get(0i16);

    //~ MONO_ITEM fn Struct::<Struct<&str>>::new
    //~ MONO_ITEM fn id::<Struct<&str>>
    let _ = (Struct::new(Struct::new("str")).f)(Struct::new("str"));

    0
}
