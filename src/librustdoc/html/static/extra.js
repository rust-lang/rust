// Copyright 2014-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*jslint browser: true, es5: true */
/*globals $: true, rootPath: true */

document.addEventListener('DOMContentLoaded', function() {
    'use strict';

    if (!window.playgroundUrl) {
        var runButtons = document.querySelectorAll(".test-arrow");

        for (var i = 0; i < runButtons.length; i++) {
            runButtons[i].classList.remove("test-arrow");
        }
        return;
    }
});
