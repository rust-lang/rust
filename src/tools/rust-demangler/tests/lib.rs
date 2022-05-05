use rust_demangler::*;

const MANGLED_INPUT: &str = r"
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

const DEMANGLED_OUTPUT: &str = r"
123foo[0]::bar
utf8_idents[317d481089b8c8fe]::საჭმელად_გემრიელი_სადილი
cc[4d6468d6c9fd4bb3]::spawn::{closure#0}::{closure#0}
<core[846817f741e54dfd]::slice::Iter<u8> as core[846817f741e54dfd]::iter::iterator::Iterator>::rposition::<core[846817f741e54dfd]::slice::memchr::memrchr::{closure#1}>::{closure#0}
alloc[f15a878b47eb696b]::alloc::box_free::<dyn alloc[f15a878b47eb696b]::boxed::FnBox<(), Output = ()>>
INtC8arrayvec8ArrayVechKj7b_E
<const_generic[317d481089b8c8fe]::Unsigned<11u8>>
<const_generic[317d481089b8c8fe]::Signed<152i16>>
<const_generic[317d481089b8c8fe]::Signed<-11i8>>
<const_generic[317d481089b8c8fe]::Bool<false>>
<const_generic[317d481089b8c8fe]::Bool<true>>
<const_generic[317d481089b8c8fe]::Char<'v'>>
<const_generic[317d481089b8c8fe]::Char<'\n'>>
<const_generic[317d481089b8c8fe]::Char<'∂'>>
<const_generic[317d481089b8c8fe]::Foo<_>>::foo::FOO
foo[0]
foo[0]
backtrace[0]::foo
rand[693ea8e72247470f]::rngs::adapter::reseeding::fork::FORK_HANDLER_REGISTERED.0.0
";

const DEMANGLED_OUTPUT_NO_CRATE_DISAMBIGUATORS: &str = r"
123foo[0]::bar
utf8_idents::საჭმელად_გემრიელი_სადილი
cc::spawn::{closure#0}::{closure#0}
<core::slice::Iter<u8> as core::iter::iterator::Iterator>::rposition::<core::slice::memchr::memrchr::{closure#1}>::{closure#0}
alloc::alloc::box_free::<dyn alloc::boxed::FnBox<(), Output = ()>>
INtC8arrayvec8ArrayVechKj7b_E
<const_generic::Unsigned<11u8>>
<const_generic::Signed<152i16>>
<const_generic::Signed<-11i8>>
<const_generic::Bool<false>>
<const_generic::Bool<true>>
<const_generic::Char<'v'>>
<const_generic::Char<'\n'>>
<const_generic::Char<'∂'>>
<const_generic::Foo<_>>::foo::FOO
foo[0]
foo[0]
backtrace[0]::foo
rand::rngs::adapter::reseeding::fork::FORK_HANDLER_REGISTERED.0.0
";

#[test]
fn test_demangle_lines() {
    let demangled_lines = demangle_lines(MANGLED_INPUT.lines(), None);
    for (expected, actual) in DEMANGLED_OUTPUT.lines().zip(demangled_lines) {
        assert_eq!(expected, actual);
    }
}

#[test]
fn test_demangle_lines_no_crate_disambiguators() {
    let demangled_lines = demangle_lines(MANGLED_INPUT.lines(), Some(create_disambiguator_re()));
    for (expected, actual) in DEMANGLED_OUTPUT_NO_CRATE_DISAMBIGUATORS.lines().zip(demangled_lines)
    {
        assert_eq!(expected, actual);
    }
}
