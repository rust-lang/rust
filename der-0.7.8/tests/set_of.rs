//! `SetOf` tests.

#![cfg(feature = "alloc")]

use der::{asn1::SetOfVec, DerOrd};
use proptest::{prelude::*, string::*};
use std::collections::BTreeSet;

proptest! {
    #[test]
    fn sort_equiv(bytes in bytes_regex(".{0,64}").unwrap()) {
        let mut uniq = BTreeSet::new();

        // Ensure there are no duplicates
        if bytes.iter().copied().all(move |x| uniq.insert(x)) {
            let mut expected = bytes.clone();
            expected.sort_by(|a, b| a.der_cmp(b).unwrap());

            let set = SetOfVec::try_from(bytes).unwrap();
            prop_assert_eq!(expected.as_slice(), set.as_slice());
        }
    }
}

/// Set ordering tests.
#[cfg(all(feature = "derive", feature = "oid"))]
mod ordering {
    use der::{
        asn1::{AnyRef, ObjectIdentifier, SetOf, SetOfVec},
        Decode, Sequence, ValueOrd,
    };
    use hex_literal::hex;

    /// X.501 `AttributeTypeAndValue`
    #[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence, ValueOrd)]
    pub struct AttributeTypeAndValue<'a> {
        pub oid: ObjectIdentifier,
        pub value: AnyRef<'a>,
    }

    const OUT_OF_ORDER_RDN_EXAMPLE: &[u8] =
        &hex!("311F301106035504030C0A4A4F484E20534D495448300A060355040A0C03313233");

    /// For compatibility reasons, we allow non-canonical DER with out-of-order
    /// sets in order to match the behavior of other implementations.
    #[test]
    fn allow_out_of_order_setof() {
        assert!(SetOf::<AttributeTypeAndValue<'_>, 2>::from_der(OUT_OF_ORDER_RDN_EXAMPLE).is_ok());
    }

    /// Same as above, with `SetOfVec` instead of `SetOf`.
    #[test]
    fn allow_out_of_order_setofvec() {
        assert!(SetOfVec::<AttributeTypeAndValue<'_>>::from_der(OUT_OF_ORDER_RDN_EXAMPLE).is_ok());
    }

    /// Test to ensure ordering is handled correctly.
    #[test]
    fn ordering_regression() {
        let der_bytes = hex!("3139301906035504030C12546573742055736572393031353734333830301C060A0992268993F22C640101130E3437303031303030303134373333");
        let set = SetOf::<AttributeTypeAndValue<'_>, 3>::from_der(&der_bytes).unwrap();
        let attr1 = set.get(0).unwrap();
        assert_eq!(ObjectIdentifier::new("2.5.4.3").unwrap(), attr1.oid);
    }
}
