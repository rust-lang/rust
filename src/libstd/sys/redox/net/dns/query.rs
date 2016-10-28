use string::String;

#[derive(Clone, Debug)]
pub struct DnsQuery {
    pub name: String,
    pub q_type: u16,
    pub q_class: u16
}
