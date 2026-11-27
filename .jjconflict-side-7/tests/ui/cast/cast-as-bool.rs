//@ dont-require-annotations: SUGGESTION

fn main() {
    let u = 5 as bool; //~ ERROR cannot cast `i32` as `bool`
                       //~| HELP compare with zero instead
                       //~| SUGGESTION != 0

    let t = (1 + 2) as bool; //~ ERROR cannot cast `i32` as `bool`
                             //~| HELP compare with zero instead
                             //~| SUGGESTION != 0

    let _ = 5_u32 as bool; //~ ERROR cannot cast `u32` as `bool`
                           //~| HELP compare with zero instead

    let _ = 64.0_f64 as bool; //~ ERROR cannot cast `f64` as `bool`
                              //~| HELP compare with zero instead

    // Enums that can normally be cast to integers can't be cast to `bool`, just like integers.
    // Note that enums that cannot be cast to integers can't be cast to anything at *all*
    // so that's not tested here.
    enum IntEnum {
        Zero,
        One,
        Two
    }
    let _ = IntEnum::One as bool; //~ ERROR cannot cast `IntEnum` as `bool`

    fn uwu(_: u8) -> String {
        todo!()
    }

    unsafe fn owo() {}

    // fn item to bool
    let _ = uwu as bool; //~ ERROR cannot cast `fn(u8) -> String {uwu}` as `bool`
    // unsafe fn item
    let _ = owo as bool; //~ ERROR cannot cast `unsafe fn() {owo}` as `bool`

    // fn ptr to bool
    let _ = uwu as fn(u8) -> String as bool; //~ ERROR cannot cast `fn(u8) -> String` as `bool`

    let _ = 'x' as bool; //~ ERROR cannot cast `char` as `bool`

    let ptr = 1 as *const ();

    let _ = ptr as bool; //~ ERROR cannot cast `*const ()` as `bool`

    let v = "hello" as bool;
    //~^ ERROR casting `&'static str` as `bool` is invalid
    //~| HELP consider using the `is_empty` method on `&'static str` to determine if it contains anything
}
