/* Checks all four scenarios possible in report_elision_failure() of
 * rustc_resolve::late::lifetimes::LifetimeContext related to returning
 * borrowed values, in various configurations.
 */

fn f1() -> &i32 { loop {} }
//~^ ERROR missing lifetime specifier [E0106]
fn f1_() -> (&i32, &i32) { loop {} }
//~^ ERROR missing lifetime specifiers [E0106]

fn f2(a: i32, b: i32) -> &i32 { loop {} }
//~^ ERROR missing lifetime specifier [E0106]
fn f2_(a: i32, b: i32) -> (&i32, &i32) { loop {} }
//~^ ERROR missing lifetime specifiers [E0106]

struct S<'a, 'b> { a: &'a i32, b: &'b i32 }
fn f3(s: &S) -> &i32 { loop {} }
//~^ ERROR missing lifetime specifier [E0106]
fn f3_(s: &S, t: &S) -> (&i32, &i32) { loop {} }
//~^ ERROR missing lifetime specifiers [E0106]

fn f4<'a, 'b>(a: &'a i32, b: &'b i32) -> &i32 { loop {} }
//~^ ERROR missing lifetime specifier [E0106]
fn f4_<'a, 'b>(a: &'a i32, b: &'b i32) -> (&i32, &i32) { loop {} }
//~^ ERROR missing lifetime specifiers [E0106]

fn f5<'a>(a: &'a i32, b: &i32) -> &i32 { loop {} }
//~^ ERROR missing lifetime specifier [E0106]
fn f5_<'a>(a: &'a i32, b: &i32) -> (&i32, &i32) { loop {} }
//~^ ERROR missing lifetime specifiers [E0106]

fn main() {}
