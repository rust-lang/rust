// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
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
        return;
    }

    var featureRegexp = new RegExp('^\s*#!\\[feature\\(\.*?\\)\\]');
    var elements = document.querySelectorAll('pre.rust-example-rendered');

    Array.prototype.forEach.call(elements, function(el) {
        el.onmouseover = function(e) {
            if (el.contains(e.relatedTarget)) {
                return;
            }

            var a = document.createElement('a');
            a.textContent = '⇱';
            a.setAttribute('class', 'test-arrow');

            var code = el.previousElementSibling.textContent;

            var channel = '';
            if (featureRegexp.test(code)) {
                channel = '&version=nightly';
            }

            a.setAttribute('href', window.playgroundUrl + '?code=' +
                           encodeURIComponent(code) + channel);
            a.setAttribute('target', '_blank');

            el.appendChild(a);
        };

        el.onmouseout = function(e) {
            if (el.contains(e.relatedTarget)) {
                return;
            }

            el.removeChild(el.querySelectorAll('a.test-arrow')[0]);
        };
    });
});
