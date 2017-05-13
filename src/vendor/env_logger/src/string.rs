use std::fmt;

pub struct Filter {
    inner: String,
}

impl Filter {
    pub fn new(spec: &str) -> Result<Filter, String> {
        Ok(Filter { inner: spec.to_string() })
    }

    pub fn is_match(&self, s: &str) -> bool {
        s.contains(&self.inner)
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}
