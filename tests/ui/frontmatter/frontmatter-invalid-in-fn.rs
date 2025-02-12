#![feature(frontmatter)]
//~^ unknown feature `frontmatter`

fn main() {
---
---
}
//~^ ERROR expected expression, found `}`
