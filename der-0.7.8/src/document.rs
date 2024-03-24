//! ASN.1 DER-encoded documents stored on the heap.

use crate::{Decode, Encode, Error, FixedTag, Length, Reader, Result, SliceReader, Tag, Writer};
use alloc::vec::Vec;
use core::fmt::{self, Debug};

#[cfg(feature = "pem")]
use {crate::pem, alloc::string::String};

#[cfg(feature = "std")]
use std::{fs, path::Path};

#[cfg(all(feature = "pem", feature = "std"))]
use alloc::borrow::ToOwned;

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

/// ASN.1 DER-encoded document.
///
/// This type wraps an encoded ASN.1 DER message. The document checked to
/// ensure it contains a valid DER-encoded `SEQUENCE`.
///
/// It implements common functionality related to encoding/decoding such
/// documents, such as PEM encapsulation as well as reading/writing documents
/// from/to the filesystem.
///
/// The [`SecretDocument`] provides a wrapper for this type with additional
/// hardening applied.
#[derive(Clone, Eq, PartialEq)]
pub struct Document {
    /// ASN.1 DER encoded bytes.
    der_bytes: Vec<u8>,

    /// Length of this document.
    length: Length,
}

impl Document {
    /// Get the ASN.1 DER-encoded bytes of this document.
    pub fn as_bytes(&self) -> &[u8] {
        self.der_bytes.as_slice()
    }

    /// Convert to a [`SecretDocument`].
    #[cfg(feature = "zeroize")]
    pub fn into_secret(self) -> SecretDocument {
        SecretDocument(self)
    }

    /// Convert to an ASN.1 DER-encoded byte vector.
    pub fn into_vec(self) -> Vec<u8> {
        self.der_bytes
    }

    /// Return an ASN.1 DER-encoded byte vector.
    pub fn to_vec(&self) -> Vec<u8> {
        self.der_bytes.clone()
    }

    /// Get the length of the encoded ASN.1 DER in bytes.
    pub fn len(&self) -> Length {
        self.length
    }

    /// Try to decode the inner ASN.1 DER message contained in this
    /// [`Document`] as the given type.
    pub fn decode_msg<'a, T: Decode<'a>>(&'a self) -> Result<T> {
        T::from_der(self.as_bytes())
    }

    /// Encode the provided type as ASN.1 DER, storing the resulting encoded DER
    /// as a [`Document`].
    pub fn encode_msg<T: Encode>(msg: &T) -> Result<Self> {
        msg.to_der()?.try_into()
    }

    /// Decode ASN.1 DER document from PEM.
    ///
    /// Returns the PEM label and decoded [`Document`] on success.
    #[cfg(feature = "pem")]
    pub fn from_pem(pem: &str) -> Result<(&str, Self)> {
        let (label, der_bytes) = pem::decode_vec(pem.as_bytes())?;
        Ok((label, der_bytes.try_into()?))
    }

    /// Encode ASN.1 DER document as a PEM string with encapsulation boundaries
    /// containing the provided PEM type `label` (e.g. `CERTIFICATE`).
    #[cfg(feature = "pem")]
    pub fn to_pem(&self, label: &'static str, line_ending: pem::LineEnding) -> Result<String> {
        Ok(pem::encode_string(label, line_ending, self.as_bytes())?)
    }

    /// Read ASN.1 DER document from a file.
    #[cfg(feature = "std")]
    pub fn read_der_file(path: impl AsRef<Path>) -> Result<Self> {
        fs::read(path)?.try_into()
    }

    /// Write ASN.1 DER document to a file.
    #[cfg(feature = "std")]
    pub fn write_der_file(&self, path: impl AsRef<Path>) -> Result<()> {
        Ok(fs::write(path, self.as_bytes())?)
    }

    /// Read PEM-encoded ASN.1 DER document from a file.
    #[cfg(all(feature = "pem", feature = "std"))]
    pub fn read_pem_file(path: impl AsRef<Path>) -> Result<(String, Self)> {
        Self::from_pem(&fs::read_to_string(path)?).map(|(label, doc)| (label.to_owned(), doc))
    }

    /// Write PEM-encoded ASN.1 DER document to a file.
    #[cfg(all(feature = "pem", feature = "std"))]
    pub fn write_pem_file(
        &self,
        path: impl AsRef<Path>,
        label: &'static str,
        line_ending: pem::LineEnding,
    ) -> Result<()> {
        let pem = self.to_pem(label, line_ending)?;
        Ok(fs::write(path, pem.as_bytes())?)
    }
}

impl AsRef<[u8]> for Document {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Debug for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Document(")?;

        for byte in self.as_bytes() {
            write!(f, "{:02X}", byte)?;
        }

        f.write_str(")")
    }
}

impl<'a> Decode<'a> for Document {
    fn decode<R: Reader<'a>>(reader: &mut R) -> Result<Document> {
        let header = reader.peek_header()?;
        let length = (header.encoded_len()? + header.length)?;
        let bytes = reader.read_slice(length)?;

        Ok(Self {
            der_bytes: bytes.into(),
            length,
        })
    }
}

impl Encode for Document {
    fn encoded_len(&self) -> Result<Length> {
        Ok(self.len())
    }

    fn encode(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_bytes())
    }
}

impl FixedTag for Document {
    const TAG: Tag = Tag::Sequence;
}

impl TryFrom<&[u8]> for Document {
    type Error = Error;

    fn try_from(der_bytes: &[u8]) -> Result<Self> {
        Self::from_der(der_bytes)
    }
}

impl TryFrom<Vec<u8>> for Document {
    type Error = Error;

    fn try_from(der_bytes: Vec<u8>) -> Result<Self> {
        let mut decoder = SliceReader::new(&der_bytes)?;
        decode_sequence(&mut decoder)?;
        decoder.finish(())?;

        let length = der_bytes.len().try_into()?;
        Ok(Self { der_bytes, length })
    }
}

/// Secret [`Document`] type.
///
/// Useful for formats which represent potentially secret data, such as
/// cryptographic keys.
///
/// This type provides additional hardening such as ensuring that the contents
/// are zeroized-on-drop, and also using more restrictive file permissions when
/// writing files to disk.
#[cfg(feature = "zeroize")]
#[derive(Clone)]
pub struct SecretDocument(Document);

#[cfg(feature = "zeroize")]
impl SecretDocument {
    /// Borrow the inner serialized bytes of this document.
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }

    /// Return an allocated ASN.1 DER serialization as a byte vector.
    pub fn to_bytes(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(self.0.to_vec())
    }

    /// Get the length of the encoded ASN.1 DER in bytes.
    pub fn len(&self) -> Length {
        self.0.len()
    }

    /// Try to decode the inner ASN.1 DER message as the given type.
    pub fn decode_msg<'a, T: Decode<'a>>(&'a self) -> Result<T> {
        self.0.decode_msg()
    }

    /// Encode the provided type as ASN.1 DER.
    pub fn encode_msg<T: Encode>(msg: &T) -> Result<Self> {
        Document::encode_msg(msg).map(Self)
    }

    /// Decode ASN.1 DER document from PEM.
    #[cfg(feature = "pem")]
    pub fn from_pem(pem: &str) -> Result<(&str, Self)> {
        Document::from_pem(pem).map(|(label, doc)| (label, Self(doc)))
    }

    /// Encode ASN.1 DER document as a PEM string.
    #[cfg(feature = "pem")]
    pub fn to_pem(
        &self,
        label: &'static str,
        line_ending: pem::LineEnding,
    ) -> Result<Zeroizing<String>> {
        self.0.to_pem(label, line_ending).map(Zeroizing::new)
    }

    /// Read ASN.1 DER document from a file.
    #[cfg(feature = "std")]
    pub fn read_der_file(path: impl AsRef<Path>) -> Result<Self> {
        Document::read_der_file(path).map(Self)
    }

    /// Write ASN.1 DER document to a file.
    #[cfg(feature = "std")]
    pub fn write_der_file(&self, path: impl AsRef<Path>) -> Result<()> {
        write_secret_file(path, self.as_bytes())
    }

    /// Read PEM-encoded ASN.1 DER document from a file.
    #[cfg(all(feature = "pem", feature = "std"))]
    pub fn read_pem_file(path: impl AsRef<Path>) -> Result<(String, Self)> {
        Document::read_pem_file(path).map(|(label, doc)| (label, Self(doc)))
    }

    /// Write PEM-encoded ASN.1 DER document to a file.
    #[cfg(all(feature = "pem", feature = "std"))]
    pub fn write_pem_file(
        &self,
        path: impl AsRef<Path>,
        label: &'static str,
        line_ending: pem::LineEnding,
    ) -> Result<()> {
        write_secret_file(path, self.to_pem(label, line_ending)?.as_bytes())
    }
}
#[cfg(feature = "zeroize")]
impl Debug for SecretDocument {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("SecretDocument").finish_non_exhaustive()
    }
}

#[cfg(feature = "zeroize")]
impl Drop for SecretDocument {
    fn drop(&mut self) {
        self.0.der_bytes.zeroize();
    }
}

#[cfg(feature = "zeroize")]
impl From<Document> for SecretDocument {
    fn from(doc: Document) -> SecretDocument {
        SecretDocument(doc)
    }
}

#[cfg(feature = "zeroize")]
impl TryFrom<&[u8]> for SecretDocument {
    type Error = Error;

    fn try_from(der_bytes: &[u8]) -> Result<Self> {
        Document::try_from(der_bytes).map(Self)
    }
}

#[cfg(feature = "zeroize")]
impl TryFrom<Vec<u8>> for SecretDocument {
    type Error = Error;

    fn try_from(der_bytes: Vec<u8>) -> Result<Self> {
        Document::try_from(der_bytes).map(Self)
    }
}

#[cfg(feature = "zeroize")]
impl ZeroizeOnDrop for SecretDocument {}

/// Attempt to decode a ASN.1 `SEQUENCE` from the given decoder, returning the
/// entire sequence including the header.
fn decode_sequence<'a>(decoder: &mut SliceReader<'a>) -> Result<&'a [u8]> {
    let header = decoder.peek_header()?;
    header.tag.assert_eq(Tag::Sequence)?;

    let len = (header.encoded_len()? + header.length)?;
    decoder.read_slice(len)
}

/// Write a file containing secret data to the filesystem, restricting the
/// file permissions so it's only readable by the owner
#[cfg(all(unix, feature = "std", feature = "zeroize"))]
fn write_secret_file(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};

    /// File permissions for secret data
    #[cfg(unix)]
    const SECRET_FILE_PERMS: u32 = 0o600;

    fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(SECRET_FILE_PERMS)
        .open(path)
        .and_then(|mut file| file.write_all(data))?;

    Ok(())
}

/// Write a file containing secret data to the filesystem
// TODO(tarcieri): permissions hardening on Windows
#[cfg(all(not(unix), feature = "std", feature = "zeroize"))]
fn write_secret_file(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    fs::write(path, data)?;
    Ok(())
}
