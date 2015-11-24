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

function change_style(elem) {
    var objs = document.getElementById("style-changer").getElementsByTagName("div");

    for (var i = 0; i < objs.length; ++i) {
        objs[i].style.display = "";
    }
    var links = document.getElementsByTagName("link");
    for (var i = 0; i < links.length; ++i) {
        var elems = links[i].href.split("/");

        links[i].disabled = (elems[elems.length - 1].replace(".css", "") !== elem.className &&
            elems[elems.length - 1] !== "rustbook.css" &&
            elems[elems.length - 1] !== "rustdoc.css");
    }
    elem.style.display = "none";
    update_local_storage(elem.className);
}

function update_local_storage(theme) {
    if (typeof(Storage) !== "undefined") {
        localStorage.theme = theme;
    } else {
        // No Web Storage support so we do nothing
    }
}

function get_current_theme() {
    if(typeof(Storage) !== "undefined" && localStorage.theme !== undefined) {
        return localStorage.theme;
    }
    return "main";
}

function switch_style() {
    theme = get_current_theme();
    if (theme === "main") {
        return;
    }
    var objs = document.getElementById("style-changer").getElementsByTagName("div");

    for (var i = 0; i < objs.length; ++i) {
        if (objs[i].className === theme) {
            change_style(objs[i]);
            return;
        }
    }
}