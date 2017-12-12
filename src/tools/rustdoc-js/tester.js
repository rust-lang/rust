// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const fs = require('fs');

const TEST_FOLDER = 'src/test/rustdoc-js/';

function loadFile(filePath) {
    var src = fs.readFileSync(filePath, 'utf8').split('\n').slice(15, -10).join('\n');
    var Module = module.constructor;
    var m = new Module();
    m._compile(src, filePath);
    return m;
}

fs.readdirSync(TEST_FOLDER).forEach(function(file) {
    var file = require(TEST_FOLDER + file);
    const expected = file.EXPECTED;
});
