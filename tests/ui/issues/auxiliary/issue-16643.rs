#![crate_type = "lib"]

pub struct TreeBuilder<H> { pub h: H }

impl<H> TreeBuilder<H> {
    pub fn process_token(&mut self) {
        match self {
            _ => for _y in self.by_ref() {}
        }
    }
}

impl<H> Iterator for TreeBuilder<H> {
    type Item = H;

    fn next(&mut self) -> Option<H> {
        None
    }
}
