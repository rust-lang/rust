// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #21400 which itself was extracted from
// stackoverflow.com/questions/28031155/is-my-borrow-checker-drunk/28031580

fn main() {
    let mut t = Test;
    assert_eq!(t.method1("one"), Ok(1));
    assert_eq!(t.method2("two"), Ok(2));
    assert_eq!(t.test(), Ok(2));
}

struct Test;

impl Test {
    fn method1(&mut self, _arg: &str) -> Result<usize, &str> {
        Ok(1)
    }

    fn method2(self: &mut Test, _arg: &str) -> Result<usize, &str> {
        Ok(2)
    }

    fn test(self: &mut Test) -> Result<usize, &str> {
        let s = format!("abcde");
        // (Originally, this invocation of method2 was saying that `s`
        // does not live long enough.)
        let data = match self.method2(&*s) {
            Ok(r) => r,
            Err(e) => return Err(e)
        };
        Ok(data)
    }
}

// Below is a closer match for the original test that was failing to compile

pub struct GitConnect;

impl GitConnect {
    fn command(self: &mut GitConnect, _s: &str) -> Result<Vec<Vec<u8>>, &str> {
        unimplemented!()
    }

    pub fn git_upload_pack(self: &mut GitConnect) -> Result<String, &str> {
        let c = format!("git-upload-pack");

        let mut out = String::new();
        let data = self.command(&c)?;

        for line in data.iter() {
            out.push_str(&format!("{:?}", line));
        }

        Ok(out)
    }
}

