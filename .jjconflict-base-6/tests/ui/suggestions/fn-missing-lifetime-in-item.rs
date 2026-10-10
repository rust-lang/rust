struct S1<F: Fn(&i32, &i32) -> &'a i32>(F); //~ ERROR use of undeclared lifetime name `'a`
struct S2<F: Fn(&i32, &i32) -> &i32>(F); //~ ERROR missing lifetime specifier
struct S3<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
//~^ ERROR binding for associated type `Output` references lifetime `'a`, which does not appear
struct S4<F: for<'x> Fn(&'x i32, &'x i32) -> &'x i32>(F);
const C: Option<Box<dyn for<'a> Fn(&usize, &usize) -> &'a usize>> = None;
//~^ ERROR binding for associated type `Output` references lifetime `'a`, which does not appear
fn main() {}
