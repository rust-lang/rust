// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::answer::DnsAnswer;
pub use self::query::DnsQuery;

use slice;
use u16;
use string::String;
use vec::Vec;

mod answer;
mod query;

#[unstable(feature = "n16", issue="0")]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, Default)]
#[repr(packed)]
pub struct n16 {
    inner: u16
}

impl n16 {
    #[unstable(feature = "n16", issue="0")]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts((&self.inner as *const u16) as *const u8, 2) }
    }

    #[unstable(feature = "n16", issue="0")]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        n16 {
            inner: unsafe { slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len()/2)[0] }
        }
    }
}

#[unstable(feature = "n16", issue="0")]
impl From<u16> for n16 {
    fn from(value: u16) -> Self {
        n16 {
            inner: value.to_be()
        }
    }
}

#[unstable(feature = "n16", issue="0")]
impl From<n16> for u16 {
    fn from(value: n16) -> Self {
        u16::from_be(value.inner)
    }
}

#[derive(Clone, Debug)]
pub struct Dns {
    pub transaction_id: u16,
    pub flags: u16,
    pub queries: Vec<DnsQuery>,
    pub answers: Vec<DnsAnswer>
}

impl Dns {
    pub fn compile(&self) -> Vec<u8> {
        let mut data = Vec::new();

        macro_rules! push_u8 {
            ($value:expr) => {
                data.push($value);
            };
        };

        macro_rules! push_n16 {
            ($value:expr) => {
                data.extend_from_slice(n16::from($value).as_bytes());
            };
        };

        push_n16!(self.transaction_id);
        push_n16!(self.flags);
        push_n16!(self.queries.len() as u16);
        push_n16!(self.answers.len() as u16);
        push_n16!(0);
        push_n16!(0);

        for query in self.queries.iter() {
            for part in query.name.split('.') {
                push_u8!(part.len() as u8);
                data.extend_from_slice(part.as_bytes());
            }
            push_u8!(0);
            push_n16!(query.q_type);
            push_n16!(query.q_class);
        }

        data
    }

    pub fn parse(data: &[u8]) -> Result<Self, String> {
        let mut i = 0;

        macro_rules! pop_u8 {
            () => {
                {
                    i += 1;
                    if i > data.len() {
                        return Err(format!("{}: {}: pop_u8", file!(), line!()));
                    }
                    data[i - 1]
                }
            };
        };

        macro_rules! pop_n16 {
            () => {
                {
                    i += 2;
                    if i > data.len() {
                        return Err(format!("{}: {}: pop_n16", file!(), line!()));
                    }
                    u16::from(n16::from_bytes(&data[i - 2 .. i]))
                }
            };
        };

        macro_rules! pop_data {
            () => {
                {
                    let mut data = Vec::new();

                    let data_len = pop_n16!();
                    for _data_i in 0..data_len {
                        data.push(pop_u8!());
                    }

                    data
                }
            };
        };

        macro_rules! pop_name {
            () => {
                {
                    let mut name = String::new();

                    loop {
                        let name_len = pop_u8!();
                        if name_len == 0 {
                            break;
                        }
                        if ! name.is_empty() {
                            name.push('.');
                        }
                        for _name_i in 0..name_len {
                            name.push(pop_u8!() as char);
                        }
                    }

                    name
                }
            };
        };

        let transaction_id = pop_n16!();
        let flags = pop_n16!();
        let queries_len = pop_n16!();
        let answers_len = pop_n16!();
        pop_n16!();
        pop_n16!();

        let mut queries = Vec::new();
        for _query_i in 0..queries_len {
            queries.push(DnsQuery {
                name: pop_name!(),
                q_type: pop_n16!(),
                q_class: pop_n16!()
            });
        }

        let mut answers = Vec::new();
        for _answer_i in 0..answers_len {
            let name_ind = 0b11000000;
            let name_test = pop_u8!();
            i -= 1;

            answers.push(DnsAnswer {
                name: if name_test & name_ind == name_ind {
                    let name_off = pop_n16!() - ((name_ind as u16) << 8);
                    let old_i = i;
                    i = name_off as usize;
                    let name = pop_name!();
                    i = old_i;
                    name
                } else {
                    pop_name!()
                },
                a_type: pop_n16!(),
                a_class: pop_n16!(),
                ttl_a: pop_n16!(),
                ttl_b: pop_n16!(),
                data: pop_data!()
            });
        }

        Ok(Dns {
            transaction_id: transaction_id,
            flags: flags,
            queries: queries,
            answers: answers,
        })
    }
}
