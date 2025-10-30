#!/usr/bin/env -S cargo -Zscript
---
[dependencies]
clap = "4"
---

//@ check-pass

// Shebangs on a file can precede a frontmatter.

#![feature(frontmatter)]

fn main () {}
