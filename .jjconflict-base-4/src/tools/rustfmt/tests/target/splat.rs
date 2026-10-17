/// Tests that the `#[rustc_splat]` attribute is preserved by rustfmt.
/// This attribute is currently unstable, and its syntax might change in future.
/// It currently uses the default formatting for attributes.

// These snippets are mainly from rust/tests/ui/splat

// Rejected by rustc, but still needs to be formatted correctly

// Can't have rust-call and splat on the same function
trait Trait: Tuple + Sized {
    extern "rust-call" fn method(#[rustc_splat] self: Self);
}

extern "rust-call" fn f(#[rustc_splat] _: ());

fn wrong_type(#[rustc_splat] _x: u32) {}

// Can't have multiple splats in the same function
fn multi_splat_bad(#[rustc_splat] (_a, _b): (u32, i8), #[rustc_splat] (_c, _d): (u32, i8)) {}

// Multiple splats on the same argument are redundant
fn multisplat_arg_bad(
    #[rustc_splat]
    #[rustc_splat]
    (_a, _b): (u32, i8),
) {
}

fn multisplat_arg_fn_bad(
    #[rustc_splat]
    #[rustc_splat]
    (_a, _b): (u32, i8),
    #[rustc_splat] (_c, _d): (u32, i8),
) {
}

// Can't have variadic and splat on the same function
unsafe extern "C" fn splat_variadic(#[rustc_splat] (_a, _b): (u32, i8), varargs: ...) {}
unsafe extern "C" fn splat_variadic2(varargs: ..., #[rustc_splat] (_a, _b): (u32, i8)) {}

// Accepted by rustc
struct Foo;

impl Foo {
    fn method(&self, #[rustc_splat] args: impl MethodArgs) -> String {}
    fn tuple_1(#[rustc_splat] (_a,): (u32,)) {}
    fn tuple_3(#[rustc_splat] (_a, _b, _c): (u32, i32, i8)) {}
}

fn generic<T: Tuple + Debug>(#[rustc_splat] a: T) -> String {
    String::new()
}

fn splat_non_terminal_arg(#[rustc_splat] (a, b): (u32, i8), c: f64) -> (i8, f64, u32) {
    (a, b, c)
}

const X: fn(#[rustc_splat] (f32,)) = None.unwrap();

fn main() {
    struct Type<T: ?Sized>(T);

    // Rejected by rustc, but still needs to be formatted correctly
    // Closures
    (|#[rustc_splat] x: i32| {})(1);

    // Function pointer types

    // Rust-call and splat aren't allowed in the same function
    let f_: extern "rust-call" fn(#[rustc_splat] ()) = f;

    let wrong_type_: fn(#[rustc_splat] _x: u32) = wrong_type;

    let multi_splat_bad_: fn(#[rustc_splat] (u32, i8), #[rustc_splat] (u32, i8)) = multi_splat_bad;
    let multisplat_arg_bad_: fn(
        #[rustc_splat]
        #[rustc_splat]
        (u32, i8),
    ) = multisplat_arg_bad;
    let multisplat_arg_fn_bad_: fn(
        #[rustc_splat]
        #[rustc_splat]
        (u32, i8),
        #[rustc_splat] (u32, i8),
    ) = multisplat_arg_fn_bad;

    // Splat and variadic aren't allowed in the same function
    let splat_variadic_: unsafe extern "C" fn(#[rustc_splat] (u32, i8), ...) = splat_variadic;
    let splat_variadic2_: unsafe extern "C" fn(..., #[rustc_splat] (u32, i8)) = splat_variadic2;

    // Accepted by rustc
    // Function pointer types

    // Only one splatted arg
    let fn_ptr: fn(#[rustc_splat] (u32, i8)) -> String =
        generic as fn(#[rustc_splat] (u32, i8)) -> String;
    impl Type<fn(#[rustc_splat] (u8, u32))> {}

    // Leading splatted arg
    let fn_pp: *const fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32) =
        &(splat_non_terminal_arg as fn(#[rustc_splat] (u32, i8), f64) -> (i8, f64, u32));

    // Trailing splatted arg
    impl Type<*mut fn(u32, i8, #[rustc_splat] (f64,))> {}

    // Middle splatted arg
    impl Type<&fn(u32, #[rustc_splat] (i8, f32, usize), f64)> {}

    // Splats within splats
    impl Type<Box<fn(u32, #[rustc_splat] (fn(#[rustc_splat] ()), i8), f64)>> {}
}
