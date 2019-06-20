// compile-flags:--cfg yes

fn f_lt<#[cfg(yes)] 'a: 'a, #[cfg(no)] T>() {}
fn f_ty<#[cfg(no)] 'a: 'a, #[cfg(yes)] T>() {}

type FnGood = for<#[cfg(yes)] 'a, #[cfg(no)] T> fn(); // OK
type FnBad = for<#[cfg(no)] 'a, #[cfg(yes)] T> fn();
//~^ ERROR only lifetime parameters can be used in this context

type PolyGood = dyn for<#[cfg(yes)] 'a, #[cfg(no)] T> Copy; // OK
type PolyBad = dyn for<#[cfg(no)] 'a, #[cfg(yes)] T> Copy;
//~^ ERROR only lifetime parameters can be used in this context

struct WhereGood where for<#[cfg(yes)] 'a, #[cfg(no)] T> u8: Copy; // OK
struct WhereBad where for<#[cfg(no)] 'a, #[cfg(yes)] T> u8: Copy;
//~^ ERROR only lifetime parameters can be used in this context

fn f_lt_no<#[cfg_attr(no, unknown)] 'a>() {} // OK
fn f_lt_yes<#[cfg_attr(yes, unknown)] 'a>() {} //~ ERROR attribute `unknown` is currently unknown
fn f_ty_no<#[cfg_attr(no, unknown)] T>() {} // OK
fn f_ty_yes<#[cfg_attr(yes, unknown)] T>() {} //~ ERROR attribute `unknown` is currently unknown

type FnNo = for<#[cfg_attr(no, unknown)] 'a> fn(); // OK
type FnYes = for<#[cfg_attr(yes, unknown)] 'a> fn();
//~^ ERROR attribute `unknown` is currently unknown

type PolyNo = dyn for<#[cfg_attr(no, unknown)] 'a> Copy; // OK
type PolyYes = dyn for<#[cfg_attr(yes, unknown)] 'a> Copy;
//~^ ERROR attribute `unknown` is currently unknown

struct WhereNo where for<#[cfg_attr(no, unknown)] 'a> u8: Copy; // OK
struct WhereYes where for<#[cfg_attr(yes, unknown)] 'a> u8: Copy;
//~^ ERROR attribute `unknown` is currently unknown

fn main() {
    f_lt::<'static>();
    f_ty::<u8>();
}
