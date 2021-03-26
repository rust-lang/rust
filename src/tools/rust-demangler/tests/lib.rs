use rust_demangler::*;

const MANGLED_LINES: &str = r"
_RNvC6_123foo3bar
_RNqCs4fqI2P2rA04_11utf8_identsu30____7hkackfecea1cbdathfdh9hlq6y
_RNCNCNgCs6DXkGYLi8lr_2cc5spawn00B5_
_RNCINkXs25_NgCsbmNqQUJIY6D_4core5sliceINyB9_4IterhENuNgNoBb_4iter8iterator8Iterator9rpositionNCNgNpB9_6memchr7memrchrs_0E0Bb_
_RINbNbCskIICzLVDPPb_5alloc5alloc8box_freeDINbNiB4_5boxed5FnBoxuEp6OutputuEL_ECs1iopQbuBiw2_3std
INtC8arrayvec8ArrayVechKj7b_E
_RMCs4fqI2P2rA04_13const_genericINtB0_8UnsignedKhb_E
_RMCs4fqI2P2rA04_13const_genericINtB0_6SignedKs98_E
_RMCs4fqI2P2rA04_13const_genericINtB0_6SignedKanb_E
_RMCs4fqI2P2rA04_13const_genericINtB0_4BoolKb0_E
_RMCs4fqI2P2rA04_13const_genericINtB0_4BoolKb1_E
_RMCs4fqI2P2rA04_13const_genericINtB0_4CharKc76_E
_RMCs4fqI2P2rA04_13const_genericINtB0_4CharKca_E
_RMCs4fqI2P2rA04_13const_genericINtB0_4CharKc2202_E
_RNvNvMCs4fqI2P2rA04_13const_genericINtB4_3FooKpE3foo3FOO
_RC3foo.llvm.9D1C9369
_RC3foo.llvm.9D1C9369@@16
_RNvC9backtrace3foo.llvm.A5310EB9
_RNvNtNtNtNtCs92dm3009vxr_4rand4rngs7adapter9reseeding4fork23FORK_HANDLER_REGISTERED.0.0
";

#[test]
fn test_demangle_lines() {
    let demangled_lines = demangle_lines(MANGLED_LINES, None);
    let mut iter = demangled_lines.iter();
    assert_eq!("", iter.next().unwrap());
    assert_eq!("123foo[0]::bar", iter.next().unwrap());
    assert_eq!("utf8_idents[317d481089b8c8fe]::საჭმელად_გემრიელი_სადილი", iter.next().unwrap());
    assert_eq!("cc[4d6468d6c9fd4bb3]::spawn::{closure#0}::{closure#0}", iter.next().unwrap());
    assert_eq!(
        "<core[846817f741e54dfd]::slice::Iter<u8> as core[846817f741e54dfd]::iter::iterator::Iterator>::rposition::<core[846817f741e54dfd]::slice::memchr::memrchr::{closure#1}>::{closure#0}",
        iter.next().unwrap()
    );
    assert_eq!(
        "alloc[f15a878b47eb696b]::alloc::box_free::<dyn alloc[f15a878b47eb696b]::boxed::FnBox<(), Output = ()>>",
        iter.next().unwrap()
    );
    assert_eq!("INtC8arrayvec8ArrayVechKj7b_E", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Unsigned<11: u8>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Signed<152: i16>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Signed<-11: i8>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Bool<false: bool>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Bool<true: bool>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Char<'v': char>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Char<'\\n': char>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Char<'∂': char>>", iter.next().unwrap());
    assert_eq!("<const_generic[317d481089b8c8fe]::Foo<_>>::foo::FOO", iter.next().unwrap());
    assert_eq!("foo[0]", iter.next().unwrap());
    assert_eq!("foo[0]", iter.next().unwrap());
    assert_eq!("backtrace[0]::foo", iter.next().unwrap());
    assert_eq!(
        "rand[693ea8e72247470f]::rngs::adapter::reseeding::fork::FORK_HANDLER_REGISTERED.0.0",
        iter.next().unwrap()
    );
    assert_eq!("", iter.next().unwrap());
    assert!(iter.next().is_none());
}

#[test]
fn test_demangle_lines_no_crate_disambiguators() {
    let demangled_lines = demangle_lines(MANGLED_LINES, Some(create_disambiguator_re()));
    let mut iter = demangled_lines.iter();
    assert_eq!("", iter.next().unwrap());
    assert_eq!("123foo[0]::bar", iter.next().unwrap());
    assert_eq!("utf8_idents::საჭმელად_გემრიელი_სადილი", iter.next().unwrap());
    assert_eq!("cc::spawn::{closure#0}::{closure#0}", iter.next().unwrap());
    assert_eq!(
        "<core::slice::Iter<u8> as core::iter::iterator::Iterator>::rposition::<core::slice::memchr::memrchr::{closure#1}>::{closure#0}",
        iter.next().unwrap()
    );
    assert_eq!(
        "alloc::alloc::box_free::<dyn alloc::boxed::FnBox<(), Output = ()>>",
        iter.next().unwrap()
    );
    assert_eq!("INtC8arrayvec8ArrayVechKj7b_E", iter.next().unwrap());
    assert_eq!("<const_generic::Unsigned<11: u8>>", iter.next().unwrap());
    assert_eq!("<const_generic::Signed<152: i16>>", iter.next().unwrap());
    assert_eq!("<const_generic::Signed<-11: i8>>", iter.next().unwrap());
    assert_eq!("<const_generic::Bool<false: bool>>", iter.next().unwrap());
    assert_eq!("<const_generic::Bool<true: bool>>", iter.next().unwrap());
    assert_eq!("<const_generic::Char<'v': char>>", iter.next().unwrap());
    assert_eq!("<const_generic::Char<'\\n': char>>", iter.next().unwrap());
    assert_eq!("<const_generic::Char<'∂': char>>", iter.next().unwrap());
    assert_eq!("<const_generic::Foo<_>>::foo::FOO", iter.next().unwrap());
    assert_eq!("foo[0]", iter.next().unwrap());
    assert_eq!("foo[0]", iter.next().unwrap());
    assert_eq!("backtrace[0]::foo", iter.next().unwrap());
    assert_eq!(
        "rand::rngs::adapter::reseeding::fork::FORK_HANDLER_REGISTERED.0.0",
        iter.next().unwrap()
    );
    assert_eq!("", iter.next().unwrap());
    assert!(iter.next().is_none());
}
