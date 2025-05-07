//@ compile-flags:--cfg yes --check-cfg=cfg(yes,no)

fn f_lt<#[cfg(yes)] 'a: 'a, #[cfg(false)] T>() {}
fn f_ty<#[cfg(false)] 'a: 'a, #[cfg(yes)] T>() {}

type FnGood = for<#[cfg(yes)] 'a, #[cfg(false)] T> fn(); // OK
type FnBad = for<#[cfg(false)] 'a, #[cfg(yes)] T> fn();
//~^ ERROR only lifetime parameters can be used in this context

type PolyGood = dyn for<#[cfg(yes)] 'a, #[cfg(false)] T> Copy; // OK
type PolyBad = dyn for<#[cfg(false)] 'a, #[cfg(yes)] T> Copy;
//~^ ERROR only lifetime parameters can be used in this context

struct WhereGood where for<#[cfg(yes)] 'a, #[cfg(false)] T> u8: Copy; // OK
struct WhereBad where for<#[cfg(false)] 'a, #[cfg(yes)] T> u8: Copy;
//~^ ERROR only lifetime parameters can be used in this context

fn f_lt_no<#[cfg_attr(FALSE, unknown)] 'a>() {} // OK
fn f_lt_yes<#[cfg_attr(yes, unknown)] 'a>() {}
//~^ ERROR cannot find attribute `unknown` in this scope
fn f_ty_no<#[cfg_attr(FALSE, unknown)] T>() {} // OK
fn f_ty_yes<#[cfg_attr(yes, unknown)] T>() {}
//~^ ERROR cannot find attribute `unknown` in this scope

type FnNo = for<#[cfg_attr(FALSE, unknown)] 'a> fn(); // OK
type FnYes = for<#[cfg_attr(yes, unknown)] 'a> fn();
//~^ ERROR cannot find attribute `unknown` in this scope

type PolyNo = dyn for<#[cfg_attr(FALSE, unknown)] 'a> Copy; // OK
type PolyYes = dyn for<#[cfg_attr(yes, unknown)] 'a> Copy;
//~^ ERROR cannot find attribute `unknown` in this scope

struct WhereNo where for<#[cfg_attr(FALSE, unknown)] 'a> u8: Copy; // OK
struct WhereYes where for<#[cfg_attr(yes, unknown)] 'a> u8: Copy;
//~^ ERROR cannot find attribute `unknown` in this scope

fn main() {
    f_lt::<'static>();
    f_ty::<u8>();
}
