use super::*;

#[test]
fn dwarf_reader() {
    let encoded: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 0xE5, 0x8E, 0x26, 0x9B, 0xF1, 0x59, 0xFF, 0xFF];

    let mut reader = DwarfReader::new(encoded.as_ptr());

    unsafe {
        assert!(reader.read::<u8>() == u8::to_be(1u8));
        assert!(reader.read::<u16>() == u16::to_be(0x0203));
        assert!(reader.read::<u32>() == u32::to_be(0x04050607));

        assert!(reader.read_uleb128() == 624485);
        assert!(reader.read_sleb128() == -624485);

        assert!(reader.read::<i8>() == i8::to_be(-1));
    }
}
