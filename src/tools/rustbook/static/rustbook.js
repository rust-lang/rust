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

    document.getElementById('toggle-nav').onclick = function(e) {
        var toc = document.getElementById('toc');
        var pagewrapper = document.getElementById('page-wrapper');
        toggleClass(toc, 'mobile-hidden');
        toggleClass(pagewrapper, 'mobile-hidden');
    };

    function toggleClass(el, className) {
        // from http://youmightnotneedjquery.com/
        if (el.classList) {
            el.classList.toggle(className);
        } else {
            var classes = el.className.split(' ');
            var existingIndex = classes.indexOf(className);

            if (existingIndex >= 0) {
                classes.splice(existingIndex, 1);
            } else {
                classes.push(className);
            }

            el.className = classes.join(' ');
        }
    }

    // The below code is used to add prev and next navigation links to the
    // bottom of each of the sections.
    // It works by extracting the current page based on the url and iterates
    // over the menu links until it finds the menu item for the current page. We
    // then create a copy of the preceding and following menu links and add the
    // correct css class and insert them into the bottom of the page.
    var toc = document.getElementById('toc').getElementsByTagName('a');
    var href = document.location.pathname.split('/').pop();

    if (href === 'index.html' || href === '') {
        href = 'README.html';
    }

    for (var i = 0; i < toc.length; i++) {
        if (toc[i].attributes.href.value.split('/').pop() === href) {
            var nav = document.createElement('p');

            if (i > 0) {
                var prevNode = toc[i-1].cloneNode(true);
                prevNode.className = 'left';
                prevNode.setAttribute('rel', 'prev');
                nav.appendChild(prevNode);
            }

            if (i < toc.length - 1) {
                var nextNode = toc[i+1].cloneNode(true);
                nextNode.className = 'right';
                nextNode.setAttribute('rel', 'next');
                nav.appendChild(nextNode);
            }

            document.getElementById('page').appendChild(nav);

            break;
        }
    }
});
