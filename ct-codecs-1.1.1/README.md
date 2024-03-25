# CT-Codecs

A reimplementation of the base64 and hexadecimal codecs from libsodium and libhydrogen in Rust.

- Constant-time for a given length, suitable for cryptographic purposes
- Strict (base64 strings are not malleable)
- Supports padded and unpadded, original and URL-safe base64 variants
- Supports characters to be ignored by the decoder
- Zero dependencies, `no_std` friendly.

## [API documentation](https://docs.rs/ct-codecs)

## Example usage

```rust
use ct_codecs::{Base64UrlSafe, Decoder, Encoder};

let encoded = Base64UrlSafe::encode_to_string(x)?;
let decoded = Base64UrlSafe::decode_to_vec(encoded, None)?;
```

