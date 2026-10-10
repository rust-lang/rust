// A regression test for trait-system-refactor-initiative#307. Proving
// `dyn D<'a, 'a>: Tr<'whatever>` passed in HIR typeck, but after
// uniquification proving `dyn D<'x, 'y>: Tr<'z>` in MIR borrowck
// failed with ambiguity. We incorrectly ignored this ambiguous result
// without registering any constraints, causing `f` to compile.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait Tr<'t> { fn get(&self) -> &'t str; }
trait D<'a, 'b>: Tr<'a> + Tr<'b> {}
//[current]~^ ERROR: type annotations needed: cannot satisfy `Self: Tr<'a>`

impl<'t> Tr<'t> for &'t str { fn get(&self) -> &'t str { *self } }
impl<'a> D<'a, 'a> for &'a str {}

fn f<'a>(x: &dyn D<'a, 'a>) -> &'static str { x.get() }
//[current]~^ ERROR: lifetime may not live long enough
//[next]~^^ ERROR: type annotations needed: cannot satisfy `dyn D<'_, '_>: Tr<'_>`

fn main() {
    let b = String::from("hi");
    let r: &'static str = f(&b.as_str());
    drop(b);
    println!("{r}");
}