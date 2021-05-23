// rustfmt-max_width: 80

struct Foo {
    #[cfg(feature = "serde")]
    bytes: [[u8; 17]; 5], // Same size as signature::ED25519_PKCS8_V2_LEN
}
