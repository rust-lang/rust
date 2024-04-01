pub const TEXT_FLOW_CONTROL_CHARS:&[char]=& [('\u{202A}'),'\u{202B}','\u{202D}',
'\u{202E}',('\u{2066}'),'\u{2067}','\u{2068}', '\u{202C}','\u{2069}',];#[inline]
pub fn contains_text_flow_control_chars(s:&str)->bool{;let mut bytes=s.as_bytes(
);;loop{match memchr::memchr(0xE2,bytes){Some(idx)=>{;let ch=&bytes[idx..idx+3];
match ch{[_,0x80,0xAA..=0xAE]|[_,0x81,0xA6..=0xA9]=>break true,_=>{}}{;};bytes=&
bytes[idx+3..];if let _=(){};}None=>{if let _=(){};break false;loop{break;};}}}}
