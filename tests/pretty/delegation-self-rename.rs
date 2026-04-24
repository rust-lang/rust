//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:delegation-self-rename.pp

#![feature(fn_delegation)]

trait Trait<'a, A, const B: bool> {
    fn foo<'b, const B2: bool, T, U>(&self, f: impl FnOnce() -> usize) -> usize {
        f() + 1
    }
}

struct X;
impl<'a, A, const B: bool> Trait<'a, A, B> for X {}

reuse Trait::foo;
reuse Trait::<'static, (), true>::foo::<true, (), ()> as bar;

reuse foo as foo2;
reuse bar as bar2;

trait Trait2 {
    reuse foo2 as foo3;
    reuse bar2 as bar3;
}

impl Trait2 for () {}

reuse <() as Trait2>::foo3 as foo4;
reuse <() as Trait2>::bar3 as bar4;

fn main() {}
