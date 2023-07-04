fn issue1468() {
euc_jp_decoder_functions!({
let trail_minus_offset = byte.wrapping_sub(0xA1);
// Fast-track Hiragana (60% according to Lunde)
// and Katakana (10% according to Lunde).
if jis0208_lead_minus_offset == 0x03 &&
trail_minus_offset < 0x53 {
// Hiragana
handle.write_upper_bmp(0x3041 + trail_minus_offset as u16)
} else if jis0208_lead_minus_offset == 0x04 &&
trail_minus_offset < 0x56 {
// Katakana
handle.write_upper_bmp(0x30A1 + trail_minus_offset as u16)
} else if trail_minus_offset > (0xFE - 0xA1) {
if byte < 0x80 {
return (DecoderResult::Malformed(1, 0),
unread_handle_trail.unread(),
handle.written());
}
return (DecoderResult::Malformed(2, 0),
unread_handle_trail.consumed(),
handle.written());
} else {
unreachable!();
}
});
}
