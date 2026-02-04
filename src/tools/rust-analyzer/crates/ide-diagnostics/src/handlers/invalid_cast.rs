use hir::{CastError, ClosureStyle, HirDisplay};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

macro_rules! format_ty {
    ($ctx:expr, $fmt:literal, $($arg:expr),* $(,)?) => {{
        std::format!(
            $fmt,
            $(
                $arg
                    .display($ctx.sema.db, $ctx.display_target)
                    .with_closure_style(ClosureStyle::ClosureWithId)
            ),*
        )
    }}
}

// Diagnostic: invalid-cast
//
// This diagnostic is triggered if the code contains an illegal cast
pub(crate) fn invalid_cast(ctx: &DiagnosticsContext<'_>, d: &hir::InvalidCast<'_>) -> Diagnostic {
    let display_range = ctx.sema.diagnostics_display_range(d.expr.map(|it| it.into()));
    let (code, message) = match d.error {
        CastError::CastToBool => (
            DiagnosticCode::RustcHardError("E0054"),
            format_ty!(ctx, "cannot cast `{}` as `bool`", d.expr_ty),
        ),
        CastError::CastToChar => (
            DiagnosticCode::RustcHardError("E0604"),
            format_ty!(ctx, "only `u8` can be cast as `char`, not {}", d.expr_ty),
        ),
        CastError::DifferingKinds => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(
                ctx,
                "casting `{}` as `{}` is invalid: vtable kinds may not match",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::SizedUnsizedCast => (
            DiagnosticCode::RustcHardError("E0607"),
            format_ty!(
                ctx,
                "cannot cast thin pointer `{}` to fat pointer `{}`",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::Unknown | CastError::IllegalCast => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(ctx, "casting `{}` as `{}` is invalid", d.expr_ty, d.cast_ty),
        ),
        CastError::IntToWideCast => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(ctx, "cannot cast `{}` to a fat pointer `{}`", d.expr_ty, d.cast_ty),
        ),
        CastError::NeedDeref => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(
                ctx,
                "casting `{}` as `{}` is invalid: needs dereference or removal of unneeded borrow",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::NeedViaPtr => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(
                ctx,
                "casting `{}` as `{}` is invalid: needs casting through a raw pointer first",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::NeedViaThinPtr => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(
                ctx,
                "casting `{}` as `{}` is invalid: needs casting through a thin pointer first",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::NeedViaInt => (
            DiagnosticCode::RustcHardError("E0606"),
            format_ty!(
                ctx,
                "casting `{}` as `{}` is invalid: needs casting through an integer first",
                d.expr_ty,
                d.cast_ty
            ),
        ),
        CastError::NonScalar => (
            DiagnosticCode::RustcHardError("E0605"),
            format_ty!(ctx, "non-primitive cast: `{}` as `{}`", d.expr_ty, d.cast_ty),
        ),
        CastError::PtrPtrAddingAutoTraits => (
            DiagnosticCode::RustcHardError("E0804"),
            "cannot add auto trait to dyn bound via pointer cast".to_owned(),
        ),
        // CastError::UnknownCastPtrKind | CastError::UnknownExprPtrKind => (
        //     DiagnosticCode::RustcHardError("E0641"),
        //     "cannot cast to a pointer of an unknown kind".to_owned(),
        // ),
    };
    Diagnostic::new(code, message, display_range).stable()
}

// Diagnostic: cast-to-unsized
//
// This diagnostic is triggered when casting to an unsized type
pub(crate) fn cast_to_unsized(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::CastToUnsized<'_>,
) -> Diagnostic {
    let display_range = ctx.sema.diagnostics_display_range(d.expr.map(|it| it.into()));
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0620"),
        format_ty!(ctx, "cast to unsized type: `{}`", d.cast_ty),
        display_range,
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_diagnostics_with_disabled};

    #[test]
    fn cast_as_bool() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let u = 5 as bool;
          //^^^^^^^^^ error: cannot cast `i32` as `bool`

    let t = (1 + 2) as bool;
          //^^^^^^^^^^^^^^^ error: cannot cast `i32` as `bool`

    let _ = 5_u32 as bool;
          //^^^^^^^^^^^^^ error: cannot cast `u32` as `bool`

    let _ = 64.0_f64 as bool;
          //^^^^^^^^^^^^^^^^ error: cannot cast `f64` as `bool`

    enum IntEnum {
        Zero,
        One,
        Two
    }
    let _ = IntEnum::One as bool;
          //^^^^^^^^^^^^^^^^^^^^ error: cannot cast `IntEnum` as `bool`

    fn uwu(_: u8) -> i32 {
        5
    }

    unsafe fn owo() {}

    let _ = uwu as bool;
          //^^^^^^^^^^^ error: cannot cast `fn uwu(u8) -> i32` as `bool`
    let _ = owo as bool;
          //^^^^^^^^^^^ error: cannot cast `unsafe fn owo()` as `bool`

    let _ = uwu as fn(u8) -> i32 as bool;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: cannot cast `fn(u8) -> i32` as `bool`
    let _ = 'x' as bool;
          //^^^^^^^^^^^ error: cannot cast `char` as `bool`

    let ptr = 1 as *const ();

    let _ = ptr as bool;
          //^^^^^^^^^^^ error: cannot cast `*const ()` as `bool`
    let v = "hello" as bool;
          //^^^^^^^^^^^^^^^ error: casting `&'static str` as `bool` is invalid: needs casting through a raw pointer first
}
"#,
        );
    }

    #[test]
    fn cast_pointee_projection() {
        check_diagnostics(
            r#"
//- minicore: sized
trait Tag<'a> {
    type Type: ?Sized;
}

trait IntoRaw: for<'a> Tag<'a> {
    fn into_raw(this: *const <Self as Tag<'_>>::Type) -> *mut <Self as Tag<'_>>::Type;
}

impl<T: for<'a> Tag<'a>> IntoRaw for T {
    fn into_raw(this: *const <Self as Tag<'_>>::Type) -> *mut <Self as Tag<'_>>::Type {
        this as *mut T::Type
    }
}

fn main() {}
"#,
        );
    }

    #[test]
    fn cast_region_to_int() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let x: isize = 3;
    let _ = &x as *const isize as usize;
}
"#,
        );
    }

    #[test]
    fn cast_to_bare_fn() {
        check_diagnostics(
            r#"
//- minicore: sized
fn foo(_x: isize) { }

fn main() {
    let v: u64 = 5;
    let x = foo as extern "C" fn() -> isize;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `fn foo(isize)` as `fn() -> isize`
    let y = v as extern "Rust" fn(isize) -> (isize, isize);
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `u64` as `fn(isize) -> (isize, isize)`
    y(x());
}
"#,
        );
    }

    #[test]
    fn cast_to_unit() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let _ = 0u32 as ();
          //^^^^^^^^^^ error: non-primitive cast: `u32` as `()`
}
"#,
        );
    }

    #[test]
    fn cast_to_slice() {
        check_diagnostics_with_disabled(
            r#"
//- minicore: sized
fn as_bytes(_: &str) -> &[u8] {
    loop {}
}

fn main() {
    as_bytes("example") as [char];
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: cast to unsized type: `[char]`

    let arr: &[u8] = &[0, 2, 3];
    arr as [char];
  //^^^^^^^^^^^^^ error: cast to unsized type: `[char]`
}
"#,
            &["E0308"],
        );
    }

    #[test]
    fn cast() {
        check_diagnostics(
            r#"
//- minicore: sized
fn null_mut<T: ?Sized>() -> *mut T {
    loop {}
}

pub fn main() {
    let i: isize = 'Q' as isize;
    let _u: u32 = i as u32;

    // Test that `_` is correctly inferred.
    let x = &"hello";
    let mut y = x as *const _;
    y = null_mut();
}
"#,
        );
    }

    #[test]
    fn dyn_tail_need_normalization() {
        check_diagnostics(
            r#"
//- minicore: dispatch_from_dyn
trait Trait {
    type Associated;
}

impl Trait for i32 {
    type Associated = i64;
}

trait Generic<T> {}

type TraitObject = dyn Generic<<i32 as Trait>::Associated>;

struct Wrap(TraitObject);

fn cast(x: *mut TraitObject) {
    x as *mut Wrap;
}
"#,
        );
    }

    #[test]
    fn enum_to_numeric_cast() {
        check_diagnostics(
            r#"
//- minicore: sized
pub enum UnitOnly {
    Foo,
    Bar,
    Baz,
}

pub enum Fieldless {
    Tuple(),
    Struct{},
    Unit,
}

pub enum NotUnitOnlyOrFieldless {
    Foo,
    Bar(u8),
    Baz
}

fn main() {
    let unit_only = UnitOnly::Foo;

    let _ = unit_only as isize;
    let _ = unit_only as i32;
    let _ = unit_only as usize;
    let _ = unit_only as u32;


    let fieldless = Fieldless::Struct{};

    let _ = fieldless as isize;
    let _ = fieldless as i32;
    let _ = fieldless as usize;
    let _ = fieldless as u32;


    let not_unit_only_or_fieldless = NotUnitOnlyOrFieldless::Foo;

    let _ = not_unit_only_or_fieldless as isize;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `NotUnitOnlyOrFieldless` as `isize`
    let _ = not_unit_only_or_fieldless as i32;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `NotUnitOnlyOrFieldless` as `i32`
    let _ = not_unit_only_or_fieldless as usize;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `NotUnitOnlyOrFieldless` as `usize`
    let _ = not_unit_only_or_fieldless as u32;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `NotUnitOnlyOrFieldless` as `u32`
}
"#,
        );
    }

    #[test]
    fn fat_ptr_cast() {
        check_diagnostics_with_disabled(
            r#"
//- minicore: sized
trait Foo {
    fn foo(&self) {} //~ WARN method `foo` is never used
}

struct Bar;

impl Foo for Bar {}

fn to_raw<T>(_: *mut T) -> *mut () {
    loop {}
}

fn main() {
    // Test we can turn a fat pointer to array back into a thin pointer.
    let a: *const [i32] = &[1, 2, 3];
    let b = a as *const [i32; 2];

    // Test conversion to an address (usize).
    let a: *const [i32; 3] = &[1, 2, 3];
    let b: *const [i32] = a;

    // And conversion to a void pointer/address for trait objects too.
    let a: *mut dyn Foo = &mut Bar;
    let b = a as *mut () as usize;
    let c = a as *const () as usize;
    let d = to_raw(a) as usize;
}
"#,
            &["E0308"],
        );

        check_diagnostics_with_disabled(
            r#"
//- minicore: sized
trait Trait {}

struct Box<T: ?Sized>;

impl<T: ?Sized> Box<T> {
    fn new(_: T) -> Self {
        loop {}
    }
}

fn as_ptr(_: &[i32]) -> *const i32 {
    loop {}
}

fn main() {
    let a: &[i32] = &[1, 2, 3];
    let b: Box<[i32]> = Box::new([1, 2, 3]);
    let p = a as *const [i32];
    let q = as_ptr(a);

    a as usize;
  //^^^^^^^^^^ error: casting `&[i32]` as `usize` is invalid: needs casting through a raw pointer first
    a as isize;
  //^^^^^^^^^^ error: casting `&[i32]` as `isize` is invalid: needs casting through a raw pointer first
    a as i16;
  //^^^^^^^^ error: casting `&[i32]` as `i16` is invalid: needs casting through a raw pointer first
    a as u32;
  //^^^^^^^^ error: casting `&[i32]` as `u32` is invalid: needs casting through a raw pointer first
    b as usize;
  //^^^^^^^^^^ error: non-primitive cast: `Box<[i32]>` as `usize`
    p as usize;
  //^^^^^^^^^^ error: casting `*const [i32]` as `usize` is invalid: needs casting through a thin pointer first
    q as *const [i32];
  //^^^^^^^^^^^^^^^^^ error: cannot cast thin pointer `*const i32` to fat pointer `*const [i32]`

    let t: *mut (dyn Trait + 'static) = 0 as *mut _;
                                      //^^^^^^^^^^^ error: cannot cast `usize` to a fat pointer `*mut (dyn Trait + 'static)`

    let mut fail: *const str = 0 as *const str;
                             //^^^^^^^^^^^^^^^ error: cannot cast `usize` to a fat pointer `*const str`
    let mut fail2: *const str = 0isize as *const str;
                              //^^^^^^^^^^^^^^^^^^^^ error: cannot cast `isize` to a fat pointer `*const str`
}

fn foo<T: ?Sized>() {
    let s = 0 as *const T;
          //^^^^^^^^^^^^^ error: cannot cast `usize` to a fat pointer `*const T`
}
"#,
            &["E0308", "unused_variables"],
        );
    }

    //     #[test]
    //     fn order_dependent_cast_inference() {
    //         check_diagnostics(
    //             r#"
    // //- minicore: sized
    // fn main() {
    //     let x = &"hello";
    //     let mut y = 0 as *const _;
    //               //^^^^^^^^^^^^^ error: cannot cast to a pointer of an unknown kind
    //     y = x as *const _;
    // }
    // "#,
    //         );
    //     }

    #[test]
    fn ptr_to_ptr_different_regions() {
        check_diagnostics(
            r#"
//- minicore: sized
struct Foo<'a> { a: &'a () }

fn extend_lifetime_very_very_safely<'a>(v: *const Foo<'a>) -> *const Foo<'static> {
    // This should pass because raw pointer casts can do anything they want.
    v as *const Foo<'static>
}

trait Trait {}

fn assert_static<'a>(ptr: *mut (dyn Trait + 'a)) -> *mut (dyn Trait + 'static) {
    ptr as _
}

fn main() {
    let unit = ();
    let foo = Foo { a: &unit };
    let _long: *const Foo<'static> = extend_lifetime_very_very_safely(&foo);
}
"#,
        );
    }

    #[test]
    fn ptr_to_trait_obj_add_auto() {
        check_diagnostics(
            r#"
//- minicore: pointee
trait Trait<'a> {}

fn add_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send) {
    x as _
}

// (to test diagnostic list formatting)
fn add_multiple_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send + Sync + Unpin) {
    x as _
}
"#,
        );
    }

    #[test]
    fn ptr_to_trait_obj_add_super_auto() {
        check_diagnostics(
            r#"
//- minicore: pointee
trait Trait: Send {}
impl Trait for () {}

fn main() {
    // This is OK: `Trait` has `Send` super trait.
    &() as *const dyn Trait as *const (dyn Trait + Send);
}
"#,
        );
    }

    #[test]
    fn ptr_to_trait_obj_ok() {
        check_diagnostics(
            r#"
//- minicore: pointee, send, sync
trait Trait<'a> {}

fn remove_auto<'a>(x: *mut (dyn Trait<'a> + Send)) -> *mut dyn Trait<'a> {
    x as _
}

fn cast_inherent_lt<'a, 'b>(x: *mut (dyn Trait<'static> + 'a)) -> *mut (dyn Trait<'static> + 'b) {
    x as _
}

fn unprincipled<'a, 'b>(x: *mut (dyn Send + 'a)) -> *mut (dyn Sync + 'b) {
    x as _
}
"#,
        );
    }

    #[ignore = "issue #18047"]
    #[test]
    fn ptr_to_trait_obj_wrap_upcast() {
        check_diagnostics(
            r#"
//- minicore: sized
trait Super {}
trait Sub: Super {}

struct Wrapper<T: ?Sized>(T);

// This cast should not compile.
// Upcasting can't work here, because we are also changing the type (`Wrapper`),
// and reinterpreting would be confusing/surprising.
// See <https://github.com/rust-lang/rust/pull/120248#discussion_r1487739518>
fn cast(ptr: *const dyn Sub) -> *const Wrapper<dyn Super> {
    ptr as _
  //^^^^^^^^ error: casting `*const dyn Sub` as `*const Wrapper<dyn Super>` is invalid: vtable kinds may not match
}
"#,
        );
    }

    #[test]
    fn supported_cast() {
        check_diagnostics(
            r#"
//- minicore: sized
pub fn main() {
    struct String;

    let f = 1_usize as *const String;

    let _ = f as isize;
    let _ = f as usize;
    let _ = f as i8;
    let _ = f as i16;
    let _ = f as i32;
    let _ = f as i64;
    let _ = f as u8;
    let _ = f as u16;
    let _ = f as u32;
    let _ = f as u64;

    let _ = 1 as isize;
    let _ = 1 as usize;
    let _ = 1 as *const String;
    let _ = 1 as i8;
    let _ = 1 as i16;
    let _ = 1 as i32;
    let _ = 1 as i64;
    let _ = 1 as u8;
    let _ = 1 as u16;
    let _ = 1 as u32;
    let _ = 1 as u64;
    let _ = 1 as f32;
    let _ = 1 as f64;

    let _ = 1_usize as isize;
    let _ = 1_usize as usize;
    let _ = 1_usize as *const String;
    let _ = 1_usize as i8;
    let _ = 1_usize as i16;
    let _ = 1_usize as i32;
    let _ = 1_usize as i64;
    let _ = 1_usize as u8;
    let _ = 1_usize as u16;
    let _ = 1_usize as u32;
    let _ = 1_usize as u64;
    let _ = 1_usize as f32;
    let _ = 1_usize as f64;

    let _ = 1i8 as isize;
    let _ = 1i8 as usize;
    let _ = 1i8 as *const String;
    let _ = 1i8 as i8;
    let _ = 1i8 as i16;
    let _ = 1i8 as i32;
    let _ = 1i8 as i64;
    let _ = 1i8 as u8;
    let _ = 1i8 as u16;
    let _ = 1i8 as u32;
    let _ = 1i8 as u64;
    let _ = 1i8 as f32;
    let _ = 1i8 as f64;

    let _ = 1u8 as isize;
    let _ = 1u8 as usize;
    let _ = 1u8 as *const String;
    let _ = 1u8 as i8;
    let _ = 1u8 as i16;
    let _ = 1u8 as i32;
    let _ = 1u8 as i64;
    let _ = 1u8 as u8;
    let _ = 1u8 as u16;
    let _ = 1u8 as u32;
    let _ = 1u8 as u64;
    let _ = 1u8 as f32;
    let _ = 1u8 as f64;

    let _ = 1i16 as isize;
    let _ = 1i16 as usize;
    let _ = 1i16 as *const String;
    let _ = 1i16 as i8;
    let _ = 1i16 as i16;
    let _ = 1i16 as i32;
    let _ = 1i16 as i64;
    let _ = 1i16 as u8;
    let _ = 1i16 as u16;
    let _ = 1i16 as u32;
    let _ = 1i16 as u64;
    let _ = 1i16 as f32;
    let _ = 1i16 as f64;

    let _ = 1u16 as isize;
    let _ = 1u16 as usize;
    let _ = 1u16 as *const String;
    let _ = 1u16 as i8;
    let _ = 1u16 as i16;
    let _ = 1u16 as i32;
    let _ = 1u16 as i64;
    let _ = 1u16 as u8;
    let _ = 1u16 as u16;
    let _ = 1u16 as u32;
    let _ = 1u16 as u64;
    let _ = 1u16 as f32;
    let _ = 1u16 as f64;

    let _ = 1i32 as isize;
    let _ = 1i32 as usize;
    let _ = 1i32 as *const String;
    let _ = 1i32 as i8;
    let _ = 1i32 as i16;
    let _ = 1i32 as i32;
    let _ = 1i32 as i64;
    let _ = 1i32 as u8;
    let _ = 1i32 as u16;
    let _ = 1i32 as u32;
    let _ = 1i32 as u64;
    let _ = 1i32 as f32;
    let _ = 1i32 as f64;

    let _ = 1u32 as isize;
    let _ = 1u32 as usize;
    let _ = 1u32 as *const String;
    let _ = 1u32 as i8;
    let _ = 1u32 as i16;
    let _ = 1u32 as i32;
    let _ = 1u32 as i64;
    let _ = 1u32 as u8;
    let _ = 1u32 as u16;
    let _ = 1u32 as u32;
    let _ = 1u32 as u64;
    let _ = 1u32 as f32;
    let _ = 1u32 as f64;

    let _ = 1i64 as isize;
    let _ = 1i64 as usize;
    let _ = 1i64 as *const String;
    let _ = 1i64 as i8;
    let _ = 1i64 as i16;
    let _ = 1i64 as i32;
    let _ = 1i64 as i64;
    let _ = 1i64 as u8;
    let _ = 1i64 as u16;
    let _ = 1i64 as u32;
    let _ = 1i64 as u64;
    let _ = 1i64 as f32;
    let _ = 1i64 as f64;

    let _ = 1u64 as isize;
    let _ = 1u64 as usize;
    let _ = 1u64 as *const String;
    let _ = 1u64 as i8;
    let _ = 1u64 as i16;
    let _ = 1u64 as i32;
    let _ = 1u64 as i64;
    let _ = 1u64 as u8;
    let _ = 1u64 as u16;
    let _ = 1u64 as u32;
    let _ = 1u64 as u64;
    let _ = 1u64 as f32;
    let _ = 1u64 as f64;

    let _ = 1u64 as isize;
    let _ = 1u64 as usize;
    let _ = 1u64 as *const String;
    let _ = 1u64 as i8;
    let _ = 1u64 as i16;
    let _ = 1u64 as i32;
    let _ = 1u64 as i64;
    let _ = 1u64 as u8;
    let _ = 1u64 as u16;
    let _ = 1u64 as u32;
    let _ = 1u64 as u64;
    let _ = 1u64 as f32;
    let _ = 1u64 as f64;

    let _ = true as isize;
    let _ = true as usize;
    let _ = true as i8;
    let _ = true as i16;
    let _ = true as i32;
    let _ = true as i64;
    let _ = true as u8;
    let _ = true as u16;
    let _ = true as u32;
    let _ = true as u64;

    let _ = 1f32 as isize;
    let _ = 1f32 as usize;
    let _ = 1f32 as i8;
    let _ = 1f32 as i16;
    let _ = 1f32 as i32;
    let _ = 1f32 as i64;
    let _ = 1f32 as u8;
    let _ = 1f32 as u16;
    let _ = 1f32 as u32;
    let _ = 1f32 as u64;
    let _ = 1f32 as f32;
    let _ = 1f32 as f64;

    let _ = 1f64 as isize;
    let _ = 1f64 as usize;
    let _ = 1f64 as i8;
    let _ = 1f64 as i16;
    let _ = 1f64 as i32;
    let _ = 1f64 as i64;
    let _ = 1f64 as u8;
    let _ = 1f64 as u16;
    let _ = 1f64 as u32;
    let _ = 1f64 as u64;
    let _ = 1f64 as f32;
    let _ = 1f64 as f64;
}
"#,
        );
    }

    #[test]
    fn unsized_struct_cast() {
        check_diagnostics(
            r#"
//- minicore: sized
pub struct Data([u8]);

fn foo(x: &[u8]) {
    let _: *const Data = x as *const Data;
                       //^^^^^^^^^^^^^^^^ error: casting `&[u8]` as `*const Data` is invalid
}
"#,
        );
    }

    #[test]
    fn unsupported_cast() {
        check_diagnostics(
            r#"
//- minicore: sized
struct A;

fn main() {
    let _ = 1.0 as *const A;
          //^^^^^^^^^^^^^^^ error: casting `f64` as `*const A` is invalid
}
"#,
        );
    }

    #[test]
    fn issue_17897() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    _ = ((), ()) as ();
      //^^^^^^^^^^^^^^ error: non-primitive cast: `((), ())` as `()`
}
"#,
        );
    }

    #[test]
    fn rustc_issue_10991() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let nil = ();
    let _t = nil as usize;
           //^^^^^^^^^^^^ error: non-primitive cast: `()` as `usize`
}
"#,
        );
    }

    #[test]
    fn rustc_issue_17444() {
        check_diagnostics(
            r#"
//- minicore: sized
enum Test {
    Foo = 0
}

fn main() {
    let _x = Test::Foo as *const isize;
           //^^^^^^^^^^^^^^^^^^^^^^^^^ error: casting `Test` as `*const isize` is invalid
}
"#,
        );
    }

    #[test]
    fn rustc_issue_43825() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let error = error;
              //^^^^^ error: no such value in this scope

    0 as f32;
    0.0 as u32;
}
"#,
        );
    }

    #[test]
    fn rustc_issue_84213() {
        check_diagnostics(
            r#"
//- minicore: sized
struct Something {
    pub field: u32,
}

fn main() {
    let mut something = Something { field: 1337 };
    let _ = something.field;

    let _pointer_to_something = something as *const Something;
                              //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `Something` as `*const Something`

    let _mut_pointer_to_something = something as *mut Something;
                                  //^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `Something` as `*mut Something`
}
"#,
        );

        // Fixed
        check_diagnostics(
            r#"
//- minicore: sized
struct Something {
    pub field: u32,
}

fn main() {
    let mut something = Something { field: 1337 };
    let _ = something.field;

    let _pointer_to_something = &something as *const Something;

    let _mut_pointer_to_something = &mut something as *mut Something;
}
"#,
        );
    }

    #[test]
    fn rustc_issue_88621() {
        check_diagnostics(
            r#"
//- minicore: sized
#[repr(u8)]
enum Kind2 {
    Foo() = 1,
    Bar{} = 2,
    Baz = 3,
}

fn main() {
    let _ = Kind2::Foo() as u8;
          //^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `Kind2` as `u8`
}
"#,
        );
    }

    #[test]
    fn rustc_issue_89497() {
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let pointer: usize = &1_i32 as *const i32 as usize;
    let _reference: &'static i32 = unsafe { pointer as *const i32 as &'static i32 };
                                          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: non-primitive cast: `*const i32` as `&'static i32`
}
"#,
        );

        // Fixed
        check_diagnostics(
            r#"
//- minicore: sized
fn main() {
    let pointer: usize = &1_i32 as *const i32 as usize;
    let _reference: &'static i32 = unsafe { &*(pointer as *const i32) };
}
"#,
        );
    }

    #[test]
    fn rustc_issue_106883() {
        check_diagnostics_with_disabled(
            r#"
//- minicore: sized, deref
use core::ops::Deref;

struct Foo;

impl Deref for Foo {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &[]
    }
}

fn main() {
    let _ = "foo" as bool;
          //^^^^^^^^^^^^^ error: casting `&'static str` as `bool` is invalid: needs casting through a raw pointer first

    let _ = Foo as bool;
          //^^^^^^^^^^^ error: non-primitive cast: `Foo` as `bool`
}

fn _slice(bar: &[i32]) -> bool {
    bar as bool
  //^^^^^^^^^^^ error: casting `&[i32]` as `bool` is invalid: needs casting through a raw pointer first
}
"#,
            &["E0308"],
        );
    }

    #[test]
    fn trait_upcasting() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, dispatch_from_dyn
#![feature(trait_upcasting)]
trait Foo {}
trait Bar: Foo {}

impl dyn Bar {
    fn bar(&self) {
        _ = self as &dyn Foo;
    }
}
"#,
        );
    }

    #[test]
    fn issue_18047() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, dispatch_from_dyn
trait LocalFrom<T> {
    fn from(_: T) -> Self;
}
trait LocalInto<T> {
    fn into(self) -> T;
}

impl<T, U> LocalInto<U> for T
where
    U: LocalFrom<T>,
{
    fn into(self) -> U {
        U::from(self)
    }
}

impl<T> LocalFrom<T> for T {
    fn from(t: T) -> T {
        t
    }
}

trait Foo {
    type ErrorType;
    type Assoc;
}

trait Bar {
    type ErrorType;
}

struct ErrorLike;

impl<E> LocalFrom<E> for ErrorLike
where
    E: Trait + 'static,
{
    fn from(_: E) -> Self {
        loop {}
    }
}

trait Baz {
    type Assoc: Bar;
    type Error: LocalInto<ErrorLike>;
}

impl<T, U> Baz for T
where
    T: Foo<Assoc = U>,
    T::ErrorType: LocalInto<ErrorLike>,
    U: Bar,
    <U as Bar>::ErrorType: LocalInto<ErrorLike>,
{
    type Assoc = U;
    type Error = T::ErrorType;
}
struct S;
trait Trait {}
impl Trait for S {}

fn test<T>()
where
    T: Baz,
    T::Assoc: 'static,
{
    let _ = &S as &dyn Trait;
}
"#,
        );
    }

    #[test]
    fn cast_literal_to_char() {
        check_diagnostics(
            r#"
fn foo() {
    0 as char;
}
            "#,
        );
    }

    #[test]
    fn cast_isize_to_infer_pointer() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized
struct Foo {}

struct Wrap<'a>(&'a mut Foo);

fn main() {
    let lparam: isize = 0;

    let _wrap = Wrap(unsafe { &mut *(lparam as *mut _) });
}
        "#,
        );
    }

    #[test]
    fn regression_18682() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized
struct Flexible {
    body: [u8],
}

trait Field {
    type Type: ?Sized;
}

impl Field for Flexible {
    type Type = [u8];
}

trait KnownLayout {
    type MaybeUninit: ?Sized;
}


impl<T> KnownLayout for [T] {
    type MaybeUninit = [T];
}

struct ZerocopyKnownLayoutMaybeUninit(<<Flexible as Field>::Type as KnownLayout>::MaybeUninit);

fn test(ptr: *mut [u8]) -> *mut ZerocopyKnownLayoutMaybeUninit {
    ptr as *mut _
}
"#,
        );
    }

    #[test]
    fn regression_19431() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized
struct Dst([u8]);

struct Struct {
    body: Dst,
}

trait Field {
    type Type: ?Sized;
}

impl Field for Struct {
    type Type = Dst;
}

trait KnownLayout {
    type MaybeUninit: ?Sized;
    type PointerMetadata;
}

impl<T> KnownLayout for [T] {
    type MaybeUninit = [T];
    type PointerMetadata = usize;
}

impl KnownLayout for Dst {
    type MaybeUninit = Dst;
    type PointerMetadata = <[u8] as KnownLayout>::PointerMetadata;
}

struct ZerocopyKnownLayoutMaybeUninit(<<Struct as Field>::Type as KnownLayout>::MaybeUninit);

fn test(ptr: *mut ZerocopyKnownLayoutMaybeUninit) -> *mut <<Struct as Field>::Type as KnownLayout>::MaybeUninit {
    ptr as *mut _
}
"#,
        );
    }
}
