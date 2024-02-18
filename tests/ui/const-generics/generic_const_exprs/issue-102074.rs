//@ check-pass
// Checks that the NoopMethodCall lint doesn't call Instance::resolve on unresolved consts

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

#[derive(Debug, Clone)]
pub struct Aes128CipherKey([u8; Aes128Cipher::KEY_LEN]);

impl Aes128CipherKey {
    pub fn new(key: &[u8; Aes128Cipher::KEY_LEN]) -> Self {
        Self(key.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Aes128Cipher;

impl Aes128Cipher {
    const KEY_LEN: usize = 16;
}

fn main() {}
