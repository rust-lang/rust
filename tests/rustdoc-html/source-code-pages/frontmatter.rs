--- json
{"edition": "2024"}
---
#![feature(frontmatter)]

// Test that we highlight frontmatter as comments on source code pages.

//@ has 'src/frontmatter/frontmatter.rs.html'
//@ has - '//pre[@class="rust"]//span[@class="comment"]' \
//        '--- json {"edition": "2024"} ---'
