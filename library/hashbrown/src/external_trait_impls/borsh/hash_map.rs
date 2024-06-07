use crate::HashMap;

use borsh::{
    io::{Read, Result, Write},
    BorshDeserialize, BorshSerialize,
};

impl<K: BorshSerialize, V: BorshSerialize, S: BorshSerialize> BorshSerialize for HashMap<K, V, S> {
    fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
        // assuming hash may have some seed,
        // as borsh is supposed by default to be deterministic, need to write it down
        // if allocator is compile time, than one can just impl wrapper with zero bytes serde of it
        self.hash_builder.serialize(writer)?;
        // considering A stateless
        self.len().serialize(writer)?;
        for kv in self.iter() {
            kv.serialize(writer)?;
        }
        Ok(())
    }
}

impl<
        K: BorshDeserialize + core::hash::Hash + Eq,
        V: BorshDeserialize,
        S: BorshDeserialize + core::hash::BuildHasher,
    > BorshDeserialize for HashMap<K, V, S>
{
    fn deserialize_reader<R: Read>(reader: &mut R) -> Result<Self> {
        let hash_builder = S::deserialize_reader(reader)?;
        let len = usize::deserialize_reader(reader)?;
        let mut map = HashMap::with_capacity_and_hasher(len, hash_builder);
        for _ in 0..len {
            let (k, v) = <(K, V)>::deserialize_reader(reader)?;
            // can use raw api here to init from memory, so can do it other time
            map.insert(k, v);
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use borsh::{BorshDeserialize, BorshSerialize};
    use std::vec::Vec;

    #[derive(Default, BorshDeserialize, BorshSerialize, Clone)]
    struct NoHash;

    impl core::hash::BuildHasher for NoHash {
        type Hasher = NoHash;
        fn build_hasher(&self) -> NoHash {
            Self
        }
    }

    impl core::hash::Hasher for NoHash {
        fn finish(&self) -> u64 {
            42
        }

        fn write(&mut self, _bytes: &[u8]) {}
    }

    #[test]
    fn encdec() {
        let mut map = crate::HashMap::<_, _, NoHash>::default();
        map.insert(1, 2);
        map.insert(3, 4);
        let mut buf = Vec::new();
        map.serialize(&mut buf).unwrap();
        let original = map.clone();
        map = crate::HashMap::<_, _, NoHash>::deserialize_reader(&mut &buf[..]).unwrap();
        assert_eq!(original[&1], map[&1]);
        assert_eq!(original[&3], map[&3]);
        assert_eq!(original.len(), map.len());
    }
}
