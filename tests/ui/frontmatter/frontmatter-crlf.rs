#!/usr/bin/env -S cargo -Zscript
---
[dependencies]
clap = "4"
---

//@ check-pass
// ignore-tidy-cr

// crlf line endings should be accepted

#![feature(frontmatter)]

fn main() {}
