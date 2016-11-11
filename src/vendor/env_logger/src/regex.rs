extern crate regex;

use std::fmt;

use self::regex::Regex;

pub struct Filter {
    inner: Regex,
}

impl Filter {
    pub fn new(spec: &str) -> Result<Filter, String> {
        match Regex::new(spec){
            Ok(r) => Ok(Filter { inner: r }),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn is_match(&self, s: &str) -> bool {
        self.inner.is_match(s)
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}
