// Local js definitions:
/* global addClass, onEachLazy, removeClass, browserSupportsHistoryApi */
/* global updateLocalStorage, getVar, nonnull */


"use strict";

(function() {

const rootPath = getVar("root-path");

const NAME_OFFSET = 0;
const DIRS_OFFSET = 1;
const FILES_OFFSET = 2;

// WARNING: RUSTDOC_MOBILE_BREAKPOINT MEDIA QUERY
// If you update this line, then you also need to update the media query with the same
// warning in rustdoc.css
const RUSTDOC_MOBILE_BREAKPOINT = 700;

function closeSidebarIfMobile() {
    if (window.innerWidth < RUSTDOC_MOBILE_BREAKPOINT) {
        updateLocalStorage("source-sidebar-show", "false");
    }
}

/**
 * @param {rustdoc.Dir} elem
 * @param {HTMLElement} parent
 * @param {string} fullPath
 * @param {boolean} hasFoundFile
 *
 * @returns {boolean} - new value for hasFoundFile
 */
function createDirEntry(elem, parent, fullPath, hasFoundFile) {
    const dirEntry = document.createElement("details");
    const summary = document.createElement("summary");

    dirEntry.className = "dir-entry";

    fullPath += elem[NAME_OFFSET] + "/";

    summary.innerText = elem[NAME_OFFSET];
    dirEntry.appendChild(summary);

    const folders = document.createElement("div");
    folders.className = "folders";
    if (elem[DIRS_OFFSET]) {
        for (const dir of elem[DIRS_OFFSET]) {
            if (createDirEntry(dir, folders, fullPath, false)) {
                dirEntry.open = true;
                hasFoundFile = true;
            }
        }
    }
    dirEntry.appendChild(folders);

    const files = document.createElement("div");
    files.className = "files";
    if (elem[FILES_OFFSET]) {
        const w = window.location.href.split("#")[0];
        for (const file_text of elem[FILES_OFFSET]) {
            const file = document.createElement("a");
            file.innerText = file_text;
            file.href = rootPath + "src/" + fullPath + file_text + ".html";
            file.addEventListener("click", closeSidebarIfMobile);
            if (!hasFoundFile && w === file.href) {
                file.className = "selected";
                dirEntry.open = true;
                hasFoundFile = true;
            }
            files.appendChild(file);
        }
    }
    dirEntry.appendChild(files);
    parent.appendChild(dirEntry);
    return hasFoundFile;
}

window.rustdocCloseSourceSidebar = () => {
    removeClass(document.documentElement, "src-sidebar-expanded");
    updateLocalStorage("source-sidebar-show", "false");
};

window.rustdocShowSourceSidebar = () => {
    addClass(document.documentElement, "src-sidebar-expanded");
    updateLocalStorage("source-sidebar-show", "true");
};

window.rustdocToggleSrcSidebar = () => {
    if (document.documentElement.classList.contains("src-sidebar-expanded")) {
        window.rustdocCloseSourceSidebar();
    } else {
        window.rustdocShowSourceSidebar();
    }
};

// This function is called from "src-files.js", generated in `html/render/write_shared.rs`.
// eslint-disable-next-line no-unused-vars
/**
 * @param {string} srcIndexStr - strinified json map from crate name to dir structure
 */
function createSrcSidebar(srcIndexStr) {
    const container = nonnull(document.querySelector("nav.sidebar"));

    const sidebar = document.createElement("div");
    sidebar.id = "src-sidebar";
    const srcIndex = new Map(JSON.parse(srcIndexStr));

    let hasFoundFile = false;

    for (const [key, source] of srcIndex) {
        source[NAME_OFFSET] = key;
        hasFoundFile = createDirEntry(source, sidebar, "", hasFoundFile);
    }

    container.appendChild(sidebar);
    // Focus on the current file in the source files sidebar.
    const selected_elem = sidebar.getElementsByClassName("selected")[0];
    if (typeof selected_elem !== "undefined") {
        // @ts-expect-error
        selected_elem.focus();
    }
}

function highlightSrcLines() {
    const match = window.location.hash.match(/^#?(\d+)(?:-(\d+))?$/);
    if (!match) {
        return;
    }
    let from = parseInt(match[1], 10);
    let to = from;
    if (typeof match[2] !== "undefined") {
        to = parseInt(match[2], 10);
    }
    if (to < from) {
        const tmp = to;
        to = from;
        from = tmp;
    }
    const from_s = "" + from;
    let elem = document.getElementById(from_s);
    if (!elem) {
        return;
    }
    const x = document.getElementById(from_s);
    if (x) {
        x.scrollIntoView();
    }
    onEachLazy(document.querySelectorAll("a[data-nosnippet]"), e => {
        removeClass(e, "line-highlighted");
    });
    for (let i = from; i <= to; ++i) {
        elem = document.getElementById("" + i);
        if (!elem) {
            break;
        }
        addClass(elem, "line-highlighted");
    }
}

const handleSrcHighlight = (function() {
    let prev_line_id = 0;

    /** @type {function(string): void} */
    const set_fragment = name => {
        const x = window.scrollX,
            y = window.scrollY;
        if (browserSupportsHistoryApi()) {
            history.replaceState(null, "", "#" + name);
            highlightSrcLines();
        } else {
            location.replace("#" + name);
        }
        // Prevent jumps when selecting one or many lines
        window.scrollTo(x, y);
    };

    // @ts-expect-error
    return ev => {
        let cur_line_id = parseInt(ev.target.id, 10);
        // This event handler is attached to the entire line number column, but it should only
        // be run if one of the anchors is clicked. It also shouldn't do anything if the anchor
        // is clicked with a modifier key (to open a new browser tab).
        if (isNaN(cur_line_id) ||
            ev.ctrlKey ||
            ev.altKey ||
            ev.metaKey) {
            return;
        }
        ev.preventDefault();

        if (ev.shiftKey && prev_line_id) {
            // Swap selection if needed
            if (prev_line_id > cur_line_id) {
                const tmp = prev_line_id;
                prev_line_id = cur_line_id;
                cur_line_id = tmp;
            }

            set_fragment(prev_line_id + "-" + cur_line_id);
        } else {
            prev_line_id = cur_line_id;

            set_fragment("" + cur_line_id);
        }
    };
}());

window.addEventListener("hashchange", highlightSrcLines);

onEachLazy(document.querySelectorAll("a[data-nosnippet]"), el => {
    el.addEventListener("click", handleSrcHighlight);
});

highlightSrcLines();

window.createSrcSidebar = createSrcSidebar;
})();
