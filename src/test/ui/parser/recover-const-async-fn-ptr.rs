// edition:2018

type T3 = async fn(); //~ ERROR an `fn` pointer type cannot be `async`
type T4 = async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
type T5 = async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`

type FT3 = for<'a> async fn(); //~ ERROR an `fn` pointer type cannot be `async`
type FT4 = for<'a> async extern fn(); //~ ERROR an `fn` pointer type cannot be `async`
type FT5 = for<'a> async unsafe extern "C" fn(); //~ ERROR an `fn` pointer type cannot be `async`

fn main() {
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
