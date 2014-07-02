// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{io, str};

#[deriving(Clone)]
pub struct ExternalHtml{
    pub in_header: String,
    pub before_content: String,
    pub after_content: String
}

impl ExternalHtml {
    pub fn load(in_header: &[String], before_content: &[String], after_content: &[String])
            -> Option<ExternalHtml> {
        match (load_external_files(in_header),
               load_external_files(before_content),
               load_external_files(after_content)) {
            (Some(ih), Some(bc), Some(ac)) => Some(ExternalHtml {
                in_header: ih,
                before_content: bc,
                after_content: ac
            }),
            _ => None
        }
    }
}

pub fn load_string(input: &Path) -> io::IoResult<Option<String>> {
    let mut f = try!(io::File::open(input));
    let d = try!(f.read_to_end());
    Ok(str::from_utf8(d.as_slice()).map(|s| s.to_string()))
}

macro_rules! load_or_return {
    ($input: expr, $cant_read: expr, $not_utf8: expr) => {
        {
            let input = Path::new($input);
            match ::externalfiles::load_string(&input) {
                Err(e) => {
                    let _ = writeln!(&mut io::stderr(),
                                     "error reading `{}`: {}", input.display(), e);
                    return $cant_read;
                }
                Ok(None) => {
                    let _ = writeln!(&mut io::stderr(),
                                     "error reading `{}`: not UTF-8", input.display());
                    return $not_utf8;
                }
                Ok(Some(s)) => s
            }
        }
    }
}

pub fn load_external_files(names: &[String]) -> Option<String> {
    let mut out = String::new();
    for name in names.iter() {
        out.push_str(load_or_return!(name.as_slice(), None, None).as_slice());
        out.push_char('\n');
    }
    Some(out)
}
