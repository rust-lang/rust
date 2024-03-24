//! PEM decoding and encoding tests.

#![cfg(all(feature = "derive", feature = "oid", feature = "pem"))]

use der::{
    asn1::{BitString, ObjectIdentifier},
    pem::{LineEnding, PemLabel},
    Decode, DecodePem, EncodePem, Sequence,
};

/// Example SPKI document encoded as DER.
const SPKI_DER: &[u8] = include_bytes!("examples/spki.der");

/// Example SPKI document encoded as PEM.
const SPKI_PEM: &str = include_str!("examples/spki.pem");

/// X.509 `AlgorithmIdentifier`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)]
pub struct AlgorithmIdentifier {
    pub algorithm: ObjectIdentifier,
    // pub parameters: ... (not used in spki.pem)
}

/// X.509 `SubjectPublicKeyInfo` (SPKI) in borrowed form
#[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)]
pub struct SpkiBorrowed<'a> {
    pub algorithm: AlgorithmIdentifier,
    #[asn1(type = "BIT STRING")]
    pub subject_public_key: &'a [u8],
}

impl PemLabel for SpkiBorrowed<'_> {
    const PEM_LABEL: &'static str = "PUBLIC KEY";
}

/// X.509 `SubjectPublicKeyInfo` (SPKI) in owned form
#[derive(Clone, Debug, Eq, PartialEq, Sequence)]
pub struct SpkiOwned {
    pub algorithm: AlgorithmIdentifier,
    pub subject_public_key: BitString,
}

impl PemLabel for SpkiOwned {
    const PEM_LABEL: &'static str = "PUBLIC KEY";
}

#[test]
fn from_pem() {
    // Decode PEM to owned form.
    let pem_spki = SpkiOwned::from_pem(SPKI_PEM).unwrap();

    // Decode DER to borrowed form.
    let der_spki = SpkiBorrowed::from_der(SPKI_DER).unwrap();

    assert_eq!(pem_spki.algorithm, der_spki.algorithm);
    assert_eq!(
        pem_spki.subject_public_key.raw_bytes(),
        der_spki.subject_public_key
    );
}

#[test]
fn to_pem() {
    let spki = SpkiBorrowed::from_der(SPKI_DER).unwrap();
    let pem = spki.to_pem(LineEnding::LF).unwrap();
    assert_eq!(&pem, SPKI_PEM);
}
