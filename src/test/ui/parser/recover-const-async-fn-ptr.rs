// edition:2018

type T0 = const fn(); //~ ERROR an `fn` pointer type cannot be `const`
type T1 = const extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
type T2 = const unsafe extern fn(); //~ ERROR an `fn` pointer type cannot be `const`
type T3 = async fn(); //~ ERROR an `fn` pointer type cannot be `async`
type T4 = async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
type T5 = async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`
type T6 = const async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `const`
//~| ERROR an `fn` pointer type cannot be `async`

type FT0 = for<'a> const fn(); //~ ERROR an `fn` pointer type cannot be `const`
type FT1 = for<'a> const extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `const`
type FT2 = for<'a> const unsafe extern fn(); //~ ERROR an `fn` pointer type cannot be `const`
type FT3 = for<'a> async fn(); //~ ERROR an `fn` pointer type cannot be `async`
type FT4 = for<'a> async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
type FT5 = for<'a> async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`
type FT6 = for<'a> const async unsafe extern "C" fn();
//~^ ERROR an `fn` pointer type cannot be `const`
//~| ERROR an `fn` pointer type cannot be `async`

fn main() {
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
