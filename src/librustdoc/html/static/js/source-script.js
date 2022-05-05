/* eslint-env es6 */
/* eslint no-var: "error" */
/* eslint prefer-const: "error" */
/* eslint prefer-arrow-callback: "error" */

// From rust:
/* global search, sourcesIndex */

// Local js definitions:
/* global addClass, getCurrentValue, hasClass, onEachLazy, removeClass, browserSupportsHistoryApi */
/* global updateLocalStorage */

"use strict";

(function() {

function getCurrentFilePath() {
    const parts = window.location.pathname.split("/");
    const rootPathParts = window.rootPath.split("/");

    for (const rootPathPart of rootPathParts) {
        if (rootPathPart === "..") {
            parts.pop();
        }
    }
    let file = window.location.pathname.substring(parts.join("/").length);
    if (file.startsWith("/")) {
        file = file.substring(1);
    }
    return file.substring(0, file.length - 5);
}

function createDirEntry(elem, parent, fullPath, currentFile, hasFoundFile) {
    const name = document.createElement("div");
    name.className = "name";

    fullPath += elem["name"] + "/";

    name.onclick = () => {
        if (hasClass(this, "expand")) {
            removeClass(this, "expand");
        } else {
            addClass(this, "expand");
        }
    };
    name.innerText = elem["name"];

    const children = document.createElement("div");
    children.className = "children";
    const folders = document.createElement("div");
    folders.className = "folders";
    if (elem.dirs) {
        for (const dir of elem.dirs) {
            if (createDirEntry(dir, folders, fullPath, currentFile, hasFoundFile)) {
                addClass(name, "expand");
                hasFoundFile = true;
            }
        }
    }
    children.appendChild(folders);

    const files = document.createElement("div");
    files.className = "files";
    if (elem.files) {
        for (const file_text of elem.files) {
            const file = document.createElement("a");
            file.innerText = file_text;
            file.href = window.rootPath + "src/" + fullPath + file_text + ".html";
            if (!hasFoundFile && currentFile === fullPath + file_text) {
                file.className = "selected";
                addClass(name, "expand");
                hasFoundFile = true;
            }
            files.appendChild(file);
        }
    }
    search.fullPath = fullPath;
    children.appendChild(files);
    parent.appendChild(name);
    parent.appendChild(children);
    return hasFoundFile && currentFile.startsWith(fullPath);
}

function toggleSidebar() {
    const sidebar = document.querySelector("nav.sidebar");
    const child = this.children[0];
    if (child.innerText === ">") {
        sidebar.classList.add("expanded");
        child.innerText = "<";
        updateLocalStorage("source-sidebar-show", "true");
    } else {
        sidebar.classList.remove("expanded");
        child.innerText = ">";
        updateLocalStorage("source-sidebar-show", "false");
    }
}

function createSidebarToggle() {
    const sidebarToggle = document.createElement("div");
    sidebarToggle.id = "sidebar-toggle";
    sidebarToggle.onclick = toggleSidebar;

    const inner = document.createElement("div");

    if (getCurrentValue("source-sidebar-show") === "true") {
        inner.innerText = "<";
    } else {
        inner.innerText = ">";
    }

    sidebarToggle.appendChild(inner);
    return sidebarToggle;
}

// This function is called from "source-files.js", generated in `html/render/mod.rs`.
// eslint-disable-next-line no-unused-vars
function createSourceSidebar() {
    if (!window.rootPath.endsWith("/")) {
        window.rootPath += "/";
    }
    const container = document.querySelector("nav.sidebar");

    const sidebarToggle = createSidebarToggle();
    container.insertBefore(sidebarToggle, container.firstChild);

    const sidebar = document.createElement("div");
    sidebar.id = "source-sidebar";
    if (getCurrentValue("source-sidebar-show") !== "true") {
        container.classList.remove("expanded");
    } else {
        container.classList.add("expanded");
    }

    const currentFile = getCurrentFilePath();
    let hasFoundFile = false;

    const title = document.createElement("div");
    title.className = "title";
    title.innerText = "Files";
    sidebar.appendChild(title);
    Object.keys(sourcesIndex).forEach(key => {
        sourcesIndex[key].name = key;
        hasFoundFile = createDirEntry(sourcesIndex[key], sidebar, "",
                                      currentFile, hasFoundFile);
    });

    container.appendChild(sidebar);
    // Focus on the current file in the source files sidebar.
    const selected_elem = sidebar.getElementsByClassName("selected")[0];
    if (typeof selected_elem !== "undefined") {
        selected_elem.focus();
    }
}

const lineNumbersRegex = /^#?(\d+)(?:-(\d+))?$/;

function highlightSourceLines(match) {
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
    onEachLazy(document.getElementsByClassName("line-numbers"), e => {
        onEachLazy(e.getElementsByTagName("span"), i_e => {
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

const handleSourceHighlight = (function () {
    let prev_line_id = 0;

    const set_fragment = name => {
        const x = window.scrollX,
            y = window.scrollY;
        if (browserSupportsHistoryApi()) {
            history.replaceState(null, null, "#" + name);
            highlightSourceLines();
        } else {
            location.replace("#" + name);
        }
        // Prevent jumps when selecting one or many lines
        window.scrollTo(x, y);
    };

    return ev => {
        let cur_line_id = parseInt(ev.target.id, 10);
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
        return highlightSourceLines(match);
    }
});

onEachLazy(document.getElementsByClassName("line-numbers"), el => {
    el.addEventListener("click", handleSourceHighlight);
});

highlightSourceLines();

window.createSourceSidebar = createSourceSidebar;
})();
