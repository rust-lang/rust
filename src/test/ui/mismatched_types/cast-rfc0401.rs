fn illegal_cast<U:?Sized,V:?Sized>(u: *const U) -> *const V
{
    u as *const V //~ ERROR is invalid
}

fn illegal_cast_2<U:?Sized>(u: *const U) -> *const str
{
    u as *const str //~ ERROR is invalid
}

trait Foo { fn foo(&self) {} }
impl<T> Foo for T {}

trait Bar { fn foo(&self) {} }
impl<T> Bar for T {}

enum E {
    A, B
}

fn main()
{
    let f: f32 = 1.2;
    let v = core::ptr::null::<u8>();
    let fat_v : *const [u8] = unsafe { &*core::ptr::null::<[u8; 1]>()};
    let fat_sv : *const [i8] = unsafe { &*core::ptr::null::<[i8; 1]>()};
    let foo: &dyn Foo = &f;

    let _ = v as &u8; //~ ERROR non-primitive cast
    let _ = v as E; //~ ERROR non-primitive cast
    let _ = v as fn(); //~ ERROR non-primitive cast
    let _ = v as (u32,); //~ ERROR non-primitive cast
    let _ = Some(&v) as *const u8; //~ ERROR non-primitive cast

    let _ = v as f32; //~ ERROR is invalid
    let _ = main as f64; //~ ERROR is invalid
    let _ = &v as usize; //~ ERROR is invalid
    let _ = f as *const u8; //~ ERROR is invalid
    let _ = 3_i32 as bool; //~ ERROR cannot cast
    let _ = E::A as bool; //~ ERROR cannot cast
    let _ = 0x61u32 as char; //~ ERROR can be cast as

    let _ = false as f32; //~ ERROR is invalid
    let _ = E::A as f32; //~ ERROR is invalid
    let _ = 'a' as f32; //~ ERROR is invalid

    let _ = false as *const u8; //~ ERROR is invalid
    let _ = E::A as *const u8; //~ ERROR is invalid
    let _ = 'a' as *const u8; //~ ERROR is invalid

    let _ = 42usize as *const [u8]; //~ ERROR is invalid
    let _ = v as *const [u8]; //~ ERROR cannot cast
    let _ = fat_v as *const dyn Foo; //~ ERROR the size for values of type
    let _ = foo as *const str; //~ ERROR is invalid
    let _ = foo as *mut str; //~ ERROR is invalid
    let _ = main as *mut str; //~ ERROR is invalid
    let _ = &f as *mut f32; //~ ERROR is invalid
    let _ = &f as *const f64; //~ ERROR is invalid
    let _ = fat_sv as usize; //~ ERROR is invalid

    let a : *const str = "hello";
    let _ = a as *const dyn Foo; //~ ERROR the size for values of type

    // check no error cascade
    let _ = main.f as *const u32; //~ ERROR no field

    let cf: *const dyn Foo = &0;
    let _ = cf as *const [u16]; //~ ERROR is invalid
    let _ = cf as *const dyn Bar; //~ ERROR is invalid

    vec![0.0].iter().map(|s| s as f32).collect::<Vec<f32>>(); //~ ERROR is invalid
}
