// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use path::*;

pub trait PathLike {
    fn path_as_str<T>(&self, f: &fn(&str) -> T) -> T;
}

impl<'self> PathLike for &'self str {
    fn path_as_str<T>(&self, f: &fn(&str) -> T) -> T {
        f(*self)
    }
}

impl PathLike for Path {
    fn path_as_str<T>(&self, f: &fn(&str) -> T) -> T {
        let s = self.to_str();
        f(s)
    }
}

#[cfg(test)]
mod test {
    use path::*;
    use super::PathLike;

    #[test]
    fn path_like_smoke_test() {
        let expected = if cfg!(unix) { "/home" } else { "C:\\" };
        let path = Path(expected);
        path.path_as_str(|p| assert!(p == expected));
        path.path_as_str(|p| assert!(p == expected));
    }
}
