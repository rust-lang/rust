// From rust:
/* global search, sourcesIndex */

// Local js definitions:
/* global addClass, getCurrentValue, hasClass, onEachLazy, removeClass, searchState */
/* global updateLocalStorage */
(function() {

function getCurrentFilePath() {
    var parts = window.location.pathname.split("/");
    var rootPathParts = window.rootPath.split("/");

    for (var i = 0, len = rootPathParts.length; i < len; ++i) {
        if (rootPathParts[i] === "..") {
            parts.pop();
        }
    }
    var file = window.location.pathname.substring(parts.join("/").length);
    if (file.startsWith("/")) {
        file = file.substring(1);
    }
    return file.substring(0, file.length - 5);
}

function createDirEntry(elem, parent, fullPath, currentFile, hasFoundFile) {
    var name = document.createElement("div");
    name.className = "name";

    fullPath += elem["name"] + "/";

    name.onclick = function() {
        if (hasClass(this, "expand")) {
            removeClass(this, "expand");
        } else {
            addClass(this, "expand");
        }
    };
    name.innerText = elem["name"];

    var i, len;

    var children = document.createElement("div");
    children.className = "children";
    var folders = document.createElement("div");
    folders.className = "folders";
    if (elem.dirs) {
        for (i = 0, len = elem.dirs.length; i < len; ++i) {
            if (createDirEntry(elem.dirs[i], folders, fullPath, currentFile,
                               hasFoundFile)) {
                addClass(name, "expand");
                hasFoundFile = true;
            }
        }
    }
    children.appendChild(folders);

    var files = document.createElement("div");
    files.className = "files";
    if (elem.files) {
        for (i = 0, len = elem.files.length; i < len; ++i) {
            var file = document.createElement("a");
            file.innerText = elem.files[i];
            file.href = window.rootPath + "src/" + fullPath + elem.files[i] + ".html";
            if (!hasFoundFile && currentFile === fullPath + elem.files[i]) {
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
    var sidebar = document.querySelector("nav.sidebar");
    var child = this.children[0];
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
    var sidebarToggle = document.createElement("div");
    sidebarToggle.id = "sidebar-toggle";
    sidebarToggle.onclick = toggleSidebar;

    var inner = document.createElement("div");

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
    var container = document.querySelector("nav.sidebar");

    var sidebarToggle = createSidebarToggle();
    container.insertBefore(sidebarToggle, container.firstChild);

    var sidebar = document.createElement("div");
    sidebar.id = "source-sidebar";
    if (getCurrentValue("source-sidebar-show") !== "true") {
        container.classList.remove("expanded");
    } else {
        container.classList.add("expanded");
    }

    var currentFile = getCurrentFilePath();
    var hasFoundFile = false;

    var title = document.createElement("div");
    title.className = "title";
    title.innerText = "Files";
    sidebar.appendChild(title);
    Object.keys(sourcesIndex).forEach(function(key) {
        sourcesIndex[key].name = key;
        hasFoundFile = createDirEntry(sourcesIndex[key], sidebar, "",
                                      currentFile, hasFoundFile);
    });

    container.appendChild(sidebar);
    // Focus on the current file in the source files sidebar.
    var selected_elem = sidebar.getElementsByClassName("selected")[0];
    if (typeof selected_elem !== "undefined") {
        selected_elem.focus();
    }
}

var lineNumbersRegex = /^#?(\d+)(?:-(\d+))?$/;

function highlightSourceLines(scrollTo, match) {
    if (typeof match === "undefined") {
        match = window.location.hash.match(lineNumbersRegex);
    }
    if (!match) {
        return;
    }
    var from = parseInt(match[1], 10);
    var to = from;
    if (typeof match[2] !== "undefined") {
        to = parseInt(match[2], 10);
    }
    if (to < from) {
        var tmp = to;
        to = from;
        from = tmp;
    }
    var elem = document.getElementById(from);
    if (!elem) {
        return;
    }
    if (scrollTo) {
        var x = document.getElementById(from);
        if (x) {
            x.scrollIntoView();
        }
    }
    onEachLazy(document.getElementsByClassName("line-numbers"), function(e) {
        onEachLazy(e.getElementsByTagName("span"), function(i_e) {
            removeClass(i_e, "line-highlighted");
        });
    });
    for (var i = from; i <= to; ++i) {
        elem = document.getElementById(i);
        if (!elem) {
            break;
        }
        addClass(elem, "line-highlighted");
    }
}

var handleSourceHighlight = (function() {
    var prev_line_id = 0;

    var set_fragment = function(name) {
        var x = window.scrollX,
            y = window.scrollY;
        if (searchState.browserSupportsHistoryApi()) {
            history.replaceState(null, null, "#" + name);
            highlightSourceLines(true);
        } else {
            location.replace("#" + name);
        }
        // Prevent jumps when selecting one or many lines
        window.scrollTo(x, y);
    };

    return function(ev) {
        var cur_line_id = parseInt(ev.target.id, 10);
        ev.preventDefault();

        if (ev.shiftKey && prev_line_id) {
            // Swap selection if needed
            if (prev_line_id > cur_line_id) {
                var tmp = prev_line_id;
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

window.addEventListener("hashchange", function() {
    var match = window.location.hash.match(lineNumbersRegex);
    if (match) {
        return highlightSourceLines(false, match);
    }
});

onEachLazy(document.getElementsByClassName("line-numbers"), function(el) {
    el.addEventListener("click", handleSourceHighlight);
});

highlightSourceLines(true);

window.createSourceSidebar = createSourceSidebar;
})();
