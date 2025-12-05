// Ensure that we emit the deny-by-default lint `invalid_type_param_default` in locations where
// type parameter defaults were accidentally allowed but don't have any effect whatsoever.
//
// Tracked in <https://github.com/rust-lang/rust/issues/36887>.
// FIXME(default_type_parameter_fallback): Consider reallowing them once they work properly.

fn avg<T = i32>(_: T) {}
//~^ ERROR defaults for generic parameters are not allowed here [invalid_type_param_default]
//~| WARN this was previously accepted

// issue: <https://github.com/rust-lang/rust/issues/26812>
fn mdn<T = T::Item>(_: T) {}
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared
//~| ERROR defaults for generic parameters are not allowed here [invalid_type_param_default]
//~| WARN this was previously accepted

struct S<T>(T);
impl<T = i32> S<T> {}
//~^ ERROR defaults for generic parameters are not allowed here [invalid_type_param_default]
//~| WARN this was previously accepted

fn main() {}
