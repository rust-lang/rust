---
package.name = "" # //~ ERROR bare CR not allowed in frontmatter
package.description = "é"
---

// ignore-tidy-cr

#![feature(frontmatter)]

pub fn main() {}
