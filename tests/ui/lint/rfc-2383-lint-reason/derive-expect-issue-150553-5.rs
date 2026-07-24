// An `#[expect]` on an item is shared with the impls derived from it, but not with
// hand-written impls for the same type: lints there must still fire.

#![deny(redundant_lifetimes)]

#[derive(Clone)]
#[expect(redundant_lifetimes)]
pub struct W<'a>
where
    'a: 'static,
{
    pub r: &'a u8,
}

impl<'a> W<'a>
//~^ ERROR unnecessary lifetime parameter `'a`
where
    'a: 'static,
{
    pub fn get(&self) -> &'a u8 {
        self.r
    }
}

fn main() {}
