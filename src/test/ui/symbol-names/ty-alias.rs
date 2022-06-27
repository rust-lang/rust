// build-fail
// ignore-tidy-linelength
// revisions: legacy v0
//[legacy]compile-flags: -Z unstable-options -C symbol-mangling-version=legacy
    //[v0]compile-flags: -C symbol-mangling-version=v0

#![feature(rustc_attrs)]
#![crate_type = "lib"]

pub trait Trait<T: ?Sized> {
    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name(_ZN8ty_alias5Trait6method
    //[legacy]~| ERROR demangling(ty_alias::Trait::method::ha7
    //[legacy]~| ERROR demangling-alt(ty_alias::Trait::method)
     //[v0]~^^^^ ERROR symbol-name(_RNvYpINtCs7cCwHY5bdWN_8ty_alias5TraitpE6methodB6_)
        //[v0]~| ERROR demangling(<_ as ty_alias[53e789f59d55e7ab]::Trait<_>>::method)
        //[v0]~| ERROR demangling-alt(<_ as ty_alias::Trait<_>>::method)
    fn method() -> &'static ();
}

pub struct Foo<T>(T);
impl<T: std::ops::Deref> Trait<T::Target> for Foo<T> {
    #[rustc_symbol_name]
    //[legacy]~^ ERROR symbol-name(_ZN118_$LT$ty_alias..Foo$LT$T$GT$$u20$as$u20$ty_alias..Trait$LT$$LT$T$u20$as$u20$core..ops..deref..Deref$GT$..Target$GT$$GT$6method17
    //[legacy]~| ERROR demangling(<ty_alias::Foo<T> as ty_alias::Trait<<T as core::ops::deref::Deref>::Target>>::method::he3
    //[legacy]~| ERROR demangling-alt(<ty_alias::Foo<T> as ty_alias::Trait<<T as core::ops::deref::Deref>::Target>>::method)
     //[v0]~^^^^ ERROR symbol-name(_RNvXINICs7cCwHY5bdWN_8ty_alias0pEINtB5_3FoopEINtB5_5TraitNtYpNtNtNtCskRLgSi4TGgx_4core3ops5deref5Deref6TargetE6methodB5_)
        //[v0]~| ERROR demangling(<ty_alias[53e789f59d55e7ab]::Foo<_> as ty_alias[53e789f59d55e7ab]::Trait<<_ as core[f30d6cf379d2fe73]::ops::deref::Deref>::Target>>::method)
        //[v0]~| ERROR demangling-alt(<ty_alias::Foo<_> as ty_alias::Trait<<_ as core::ops::deref::Deref>::Target>>::method)
    fn method() -> &'static () {
        static FOO: () = ();
        &FOO
    }
}
