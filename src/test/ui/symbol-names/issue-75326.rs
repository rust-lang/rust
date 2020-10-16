// build-fail
// ignore-tidy-linelength
// revisions: legacy v0
//[legacy]compile-flags: -Z symbol-mangling-version=legacy
//[v0]compile-flags: -Z symbol-mangling-version=v0
//[legacy]normalize-stderr-32bit: "h[\d\w]+" -> "SYMBOL_HASH"
//[legacy]normalize-stderr-64bit: "h[\d\w]+" -> "SYMBOL_HASH"

#![feature(rustc_attrs)]

pub(crate) struct Foo<I, E>(I, E);

pub trait Iterator2 {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        Self: Sized,
        P: FnMut(&Self::Item) -> bool,
    {
        unimplemented!()
    }
}

struct Bar;

impl Iterator2 for Bar {
    type Item = (u32, u16);

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

impl<I, T, E> Iterator2 for Foo<I, E>
where
    I: Iterator2<Item = (T, E)>,
{
    type Item = T;

    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name(_ZN72_$LT$issue_75326..Foo$LT$I$C$E$GT$$u20$as$u20$issue_75326..Iterator2$GT$4next
    //[legacy]~| ERROR demangling(<issue_75326::Foo<I,E> as issue_75326::Iterator2>::next
    //[legacy]~| ERROR demangling-alt(<issue_75326::Foo<I,E> as issue_75326::Iterator2>::next)
    //[v0]~^^^^  ERROR symbol-name(_RNvXINICs4fqI2P2rA04_11issue_75326s_0pppEINtB5_3FooppENtB5_9Iterator24nextB5_)
    //[v0]~|     ERROR demangling(<issue_75326[317d481089b8c8fe]::Foo<_, _> as issue_75326[317d481089b8c8fe]::Iterator2>::next)
    //[v0]~|     ERROR demangling-alt(<issue_75326::Foo<_, _> as issue_75326::Iterator2>::next)
    fn next(&mut self) -> Option<Self::Item> {
        self.find(|_| true)
    }
}

fn main() {
    let mut a = Foo(Bar, 1u16);
    let _ = a.next();
}
