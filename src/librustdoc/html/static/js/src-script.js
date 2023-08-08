// From rust:
/* global srcIndex */

// Local js definitions:
/* global addClass, getCurrentValue, onEachLazy, removeClass, browserSupportsHistoryApi */
/* global updateLocalStorage, getVar */

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

function toggleSidebar() {
    const child = this.parentNode.children[0];
    if (child.innerText === ">") {
        addClass(document.documentElement, "src-sidebar-expanded");
        child.innerText = "<";
        updateLocalStorage("source-sidebar-show", "true");
    } else {
        removeClass(document.documentElement, "src-sidebar-expanded");
        child.innerText = ">";
        updateLocalStorage("source-sidebar-show", "false");
    }
}

function createSidebarToggle() {
    const sidebarToggle = document.createElement("div");
    sidebarToggle.id = "src-sidebar-toggle";

    const inner = document.createElement("button");

    if (getCurrentValue("source-sidebar-show") === "true") {
        inner.innerText = "<";
    } else {
        inner.innerText = ">";
    }
    inner.onclick = toggleSidebar;

    sidebarToggle.appendChild(inner);
    return sidebarToggle;
}

// This function is called from "src-files.js", generated in `html/render/write_shared.rs`.
// eslint-disable-next-line no-unused-vars
function createSrcSidebar() {
    const container = document.querySelector("nav.sidebar");

    const sidebarToggle = createSidebarToggle();
    container.insertBefore(sidebarToggle, container.firstChild);

    const sidebar = document.createElement("div");
    sidebar.id = "src-sidebar";

    let hasFoundFile = false;

    const title = document.createElement("div");
    title.className = "title";
    title.innerText = "Files";
    sidebar.appendChild(title);
    Object.keys(srcIndex).forEach(key => {
        srcIndex[key][NAME_OFFSET] = key;
        hasFoundFile = createDirEntry(srcIndex[key], sidebar, "", hasFoundFile);
    });

    container.appendChild(sidebar);
    // Focus on the current file in the source files sidebar.
    const selected_elem = sidebar.getElementsByClassName("selected")[0];
    if (typeof selected_elem !== "undefined") {
        selected_elem.focus();
    }
}

const lineNumbersRegex = /^#?(\d+)(?:-(\d+))?$/;

function highlightSrcLines(match) {
    if (typeof match === "undefined") {
        match = window.location.hash.match(lineNumbersRegex);
    }
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
    let elem = document.getElementById(from);
    if (!elem) {
        return;
    }
    const x = document.getElementById(from);
    if (x) {
        x.scrollIntoView();
    }
    onEachLazy(document.getElementsByClassName("src-line-numbers"), e => {
        onEachLazy(e.getElementsByTagName("a"), i_e => {
            removeClass(i_e, "line-highlighted");
        });
    });
    for (let i = from; i <= to; ++i) {
        elem = document.getElementById(i);
        if (!elem) {
            break;
        }
        addClass(elem, "line-highlighted");
    }
}

const handleSrcHighlight = (function() {
    let prev_line_id = 0;

    const set_fragment = name => {
        const x = window.scrollX,
            y = window.scrollY;
        if (browserSupportsHistoryApi()) {
            history.replaceState(null, null, "#" + name);
            highlightSrcLines();
        } else {
            location.replace("#" + name);
        }
        // Prevent jumps when selecting one or many lines
        window.scrollTo(x, y);
    };

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

            set_fragment(cur_line_id);
        }
    };
}());

window.addEventListener("hashchange", () => {
    const match = window.location.hash.match(lineNumbersRegex);
    if (match) {
        return highlightSrcLines(match);
    }
});

onEachLazy(document.getElementsByClassName("src-line-numbers"), el => {
    el.addEventListener("click", handleSrcHighlight);
});

highlightSrcLines();

window.createSrcSidebar = createSrcSidebar;
})();
