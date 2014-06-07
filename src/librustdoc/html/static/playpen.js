// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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

(function() {
    if (window.playgroundUrl) {
        $('pre.rust').hover(function() {
            var id = '#' + $(this).attr('id').replace('rendered', 'raw');
            var a = $('<a>').text('⇱').attr('class', 'test-arrow');
            var code = $(id).text();
            a.attr('href', window.playgroundUrl + '?code=' +
                           encodeURIComponent(code));
            a.attr('target', '_blank');
            $(this).append(a);
        }, function() {
            $(this).find('a.test-arrow').remove();
        });
    }
}());

