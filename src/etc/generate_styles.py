#!/usr/bin/env python
#
# Copyright 2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import os
import sys

file_data = "\
// Copyright 2013 The Rust Project Developers. See the COPYRIGHT\n\
// file at the top-level directory of this distribution and at\n\
// http://rust-lang.org/COPYRIGHT.\n\
//\n\
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or\n\
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license\n\
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your\n\
// option. This file may not be copied, modified, or distributed\n\
// except according to those terms.\n\
//\n\n\
// Generated file\n\
// Modifying it is useless\n\
//\n\n\
use std::path::PathBuf;\n\
use html::render::{write, Error};\n\n\
pub fn include_style_files(dst: &PathBuf) -> Result<(), Error> {\n"

f = open("src/librustdoc/html/styles.rs", "w")
path = "src/librustdoc/html/static/styles/"
listing = os.listdir(path)
entries = []
for entry in listing:
    if os.path.isfile(path + entry):
        entries.append(entry)
for entry in entries:
    file_data += "    try!(write(dst.join(\"{entry}\"), \
include_bytes!(\"static/styles/{entry}\")));\n".format(entry=entry)
file_data += "    Ok(())\n}\n\npub fn css_file_links(root_path: &str) -> String {\n"
file_data += "    let mut out = String::new();\n\n"

for entry in entries:
    file_data += "    out.push_str(\n        &format!(\"<link rel=\\\"stylesheet\\\" "
    file_data += "type=\\\"text/css\\\" href=\\\"{}%s\\\" \\\n                 " % entry
    if entry != "main.css":
        file_data += " disabled>\", root_path));\n"
    else:
        file_data += ">\", root_path));\n"
file_data += "    out\n}\n\npub fn get_divs() -> String {\n"
file_data += "    let mut out = String::new();\n\n"

for entry in entries:
    file_data += "    out.push_str(\"<div class=\\\""
    file_data += "{entry}\\\" onclick=\\\"change_style(this);\\\"".format(
        entry=entry.rsplit('.', 1)[0])
    if entry.rsplit('.', 1)[0] == "main":
        file_data += "\\\n                 style=\\\"display:none;\\\""
    file_data += "></div>\");\n"
file_data += "    out\n}\n"
f.write(file_data)
f.close()