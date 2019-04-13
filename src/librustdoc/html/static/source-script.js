// From rust:
/* global sourcesIndex */

// Local js definitions:
/* global addClass, getCurrentValue, hasClass, removeClass, updateLocalStorage */

function getCurrentFilePath() {
    var parts = window.location.pathname.split("/");
    var rootPathParts = window.rootPath.split("/");

    for (var i = 0; i < rootPathParts.length; ++i) {
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

    var children = document.createElement("div");
    children.className = "children";
    var folders = document.createElement("div");
    folders.className = "folders";
    if (elem.dirs) {
        for (var i = 0; i < elem.dirs.length; ++i) {
            if (createDirEntry(elem.dirs[i], folders, fullPath, currentFile,
                               hasFoundFile) === true) {
                addClass(name, "expand");
                hasFoundFile = true;
            }
        }
    }
    children.appendChild(folders);

    var files = document.createElement("div");
    files.className = "files";
    if (elem.files) {
        for (i = 0; i < elem.files.length; ++i) {
            var file = document.createElement("a");
            file.innerText = elem.files[i];
            file.href = window.rootPath + "src/" + fullPath + elem.files[i] + ".html";
            if (hasFoundFile === false &&
                    currentFile === fullPath + elem.files[i]) {
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
    return hasFoundFile === true && currentFile.startsWith(fullPath);
}

function toggleSidebar() {
    var sidebar = document.getElementById("source-sidebar");
    var child = this.children[0].children[0];
    if (child.innerText === ">") {
        sidebar.style.left = "";
        this.style.left = "";
        child.innerText = "<";
        updateLocalStorage("rustdoc-source-sidebar-show", "true");
    } else {
        sidebar.style.left = "-300px";
        this.style.left = "0";
        child.innerText = ">";
        updateLocalStorage("rustdoc-source-sidebar-show", "false");
    }
}

function createSidebarToggle() {
    var sidebarToggle = document.createElement("div");
    sidebarToggle.id = "sidebar-toggle";
    sidebarToggle.onclick = toggleSidebar;

    var inner1 = document.createElement("div");
    inner1.style.position = "relative";

    var inner2 = document.createElement("div");
    inner2.style.paddingTop = "3px";
    if (getCurrentValue("rustdoc-source-sidebar-show") === "true") {
        inner2.innerText = "<";
    } else {
        inner2.innerText = ">";
        sidebarToggle.style.left = "0";
    }

    inner1.appendChild(inner2);
    sidebarToggle.appendChild(inner1);
    return sidebarToggle;
}

function createSourceSidebar() {
    if (window.rootPath.endsWith("/") === false) {
        window.rootPath += "/";
    }
    var main = document.getElementById("main");

    var sidebarToggle = createSidebarToggle();
    main.insertBefore(sidebarToggle, main.firstChild);

    var sidebar = document.createElement("div");
    sidebar.id = "source-sidebar";
    if (getCurrentValue("rustdoc-source-sidebar-show") !== "true") {
        sidebar.style.left = "-300px";
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

    main.insertBefore(sidebar, main.firstChild);
}
