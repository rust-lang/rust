use string::String;
use vec::Vec;

#[derive(Clone, Debug)]
pub struct DnsAnswer {
    pub name: String,
    pub a_type: u16,
    pub a_class: u16,
    pub ttl_a: u16,
    pub ttl_b: u16,
    pub data: Vec<u8>
}
