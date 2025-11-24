use crate::{INLINE_CAP, Repr, SmolStr};
use alloc::string::{String, ToString};
use borsh::{
    BorshDeserialize, BorshSerialize,
    io::{Error, ErrorKind, Read, Write},
};
use core::mem::transmute;

impl BorshSerialize for SmolStr {
    fn serialize<W: Write>(&self, writer: &mut W) -> borsh::io::Result<()> {
        self.as_str().serialize(writer)
    }
}

impl BorshDeserialize for SmolStr {
    #[inline]
    fn deserialize_reader<R: Read>(reader: &mut R) -> borsh::io::Result<Self> {
        let len = u32::deserialize_reader(reader)?;
        if (len as usize) < INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            reader.read_exact(&mut buf[..len as usize])?;
            _ = core::str::from_utf8(&buf[..len as usize]).map_err(|err| {
                let msg = err.to_string();
                Error::new(ErrorKind::InvalidData, msg)
            })?;
            Ok(SmolStr(Repr::Inline {
                len: unsafe { transmute::<u8, crate::InlineSize>(len as u8) },
                buf,
            }))
        } else {
            // u8::vec_from_reader always returns Some on success in current implementation
            let vec = u8::vec_from_reader(len, reader)?
                .ok_or_else(|| Error::other("u8::vec_from_reader unexpectedly returned None"))?;
            Ok(SmolStr::from(String::from_utf8(vec).map_err(|err| {
                let msg = err.to_string();
                Error::new(ErrorKind::InvalidData, msg)
            })?))
        }
    }
}
