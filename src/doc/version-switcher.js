(function() {

let CURRENT_VERSION = -1;

function get_current_version() {
    if (CURRENT_VERSION !== -1) {
        return CURRENT_VERSION;
    }
    const now = Date.now();
    // Month is 0-indexed.
    // First release of Rust, 15 may 2015.
    const first_release = new Date(2015, 4, 15);
    const diff_time = Math.abs(now - first_release);
    const nb_days = Math.ceil(diff_time / (1000 * 60 * 60 * 24));
    const nb_weeks = nb_days / 7;
    CURRENT_VERSION = Math.floor(nb_weeks / 6);
    return CURRENT_VERSION;
}

function checkIfIsOldVersion() {
    if (["http:", "https:"].indexOf(window.location.protocol) === -1) {
        return false;
    }
    const parts = window.location.pathname.split("/");

    return parts.length > 1 && parts[1].indexOf(".") !== -1 && parts[1] !== CURRENT_VERSION;
}

function createOption(text, isDefault) {
    const option = document.createElement("option");
    option.value = text;
    option.innerText = text;
    if (isDefault) {
        option.selected = true;
    }
    return option;
}

function addStyle(css) {
    const style = document.createElement("style");
    style.type = "text/css";
    style.appendChild(document.createTextNode(css));

    document.head.appendChild(style);
}

function setupStyleFor59(rustdoc_container) {
    // nothing to do in here!
}

function setupStyleFor32(rustdoc_container, switcherEl, extraStyle) {
    document.body.style.padding = "0";
    rustdoc_container.style.position = "relative";
    rustdoc_container.style.padding = "0 15px 20px 15px";

    addStyle(`@media (min-width: 701px) {
    .rustdoc {
        padding: 10px 15px 20px 15px !important;
    }
    #switch-version-filler {
        display: block !important;
        left: 0 !important;
    }
}

.sidebar.mobile {
    top: 0 !important;
}
${extraStyle}`);

    // We also need to create a "cosmetic" element to not have a weird empty space above the
    // sidebar.
    const filler = document.createElement("div");
    filler.style.position = "fixed";
    filler.style.top = "0";
    filler.style.bottom = "0";
    filler.style.zIndex = "-1";
    filler.style.display = "none";
    filler.id = "switch-version-filler";
    document.body.appendChild(filler);

    function changeSidebarTop() {
        const height = switcherEl.getBoundingClientRect().height;
        const sidebar = document.querySelector(".sidebar");
        sidebar.style.top = height + 1 + "px";
    }
    setTimeout(() => {
        const sidebar = window.getComputedStyle(document.querySelector(".sidebar"));
        filler.style.width = sidebar.width;
        filler.style.backgroundColor = sidebar.backgroundColor;
        changeSidebarTop();
    }, 0); // it'll be computed once it's added in the DOM.
    window.addEventListener("resize", changeSidebarTop);
}

function setupStyleFor22(rustdoc_container, switcherEl) {
    // It's mostly the same as `setupStyleFor32` so we call it and make the extra changes afterward.
    setupStyleFor32(rustdoc_container, switcherEl, `@media (max-width: 700px) {
    .sidebar {
        height: 45px;
        min-height: 40px;
        margin: 0;
        margin-left: -15px;
        padding: 0 15px;
        position: static;
        z-index: 1;
    }
}`);
}

function setupStyleFor21(rustdoc_container, switcherEl) {
    // It's mostly the same as `setupStyleFor22` so we call it and make the extra changes afterward.
    document.body.style.padding = "0";

    const css = `.rustdoc {
    padding: 10px 15px 20px 15px !important;
}`;
    addStyle(css);

    function changeSidebarTop() {
        const height = switcherEl.getBoundingClientRect().height;
        const sidebar = document.querySelector(".sidebar");
        sidebar.style.top = height + 1 + "px";
    }
    setTimeout(() => {
        const sidebar = window.getComputedStyle(document.querySelector(".sidebar"));
        changeSidebarTop();
    }, 0); // it'll be computed once it's added in the DOM.
    window.addEventListener("resize", changeSidebarTop);
}

function getHtmlForSwitcher(isOldVersion, switcher_container) {
    if (!isOldVersion) {
        switcher_container.style.color = "#eee";
        return "You can pick a different version with this dropdown:&nbsp;";
    }

    switcher_container.style.color = "#e57300";

    addStyle(`#doc-version-switcher svg {
    width: 1em;
    height: 1em;
    fill: currentColor;
    padding-top: 0.1em;
}`);

    const warning_img = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512">\
<path d="M569.517 440.013C587.975 472.007 564.806 512 527.94 512H48.054c-36.937 \
0-59.999-40.055-41.577-71.987L246.423 23.985c18.467-32.009 64.72-31.951 83.154 0l239.94 \
416.028zM288 354c-25.405 0-46 20.595-46 46s20.595 46 46 46 46-20.595 \
46-46-20.595-46-46-46zm-43.673-165.346l7.418 136c.347 6.364 5.609 11.346 11.982 \
11.346h48.546c6.373 0 11.635-4.982 11.982-11.346l7.418-136c.375-6.874-5.098-12.654-\
11.982-12.654h-63.383c-6.884 0-12.356 5.78-11.981 12.654z"></path></svg>&nbsp;`;
    return warning_img + "You are seeing an outdated version of this documentation. " +
        "Click on the dropdown to go to the latest stable version:&nbsp;";
}

function showSwitcher(isOldVersion) {
    const el = document.createElement("div");

    el.style.borderBottom = "1px solid #bbb";
    el.style.fontSize = "1.1em";
    el.style.padding = "4px";
    el.style.background = "#111";
    el.style.width = "100%";
    el.id = "doc-version-switcher";

    const parts = window.location.pathname.split("/");
    parts[1] = "stable";
    const url = parts.join("/");

    const current_doc_version = window.location.pathname.split("/")[1];
    const version_picker = document.createElement("select");

    version_picker.appendChild(createOption("stable", current_doc_version === "stable"));
    version_picker.appendChild(createOption("beta", current_doc_version === "beta"));
    version_picker.appendChild(createOption("nightly", current_doc_version === "nightly"));

    for (let medium = get_current_version(); medium >= 0; --medium) {
        const version = `1.${medium}.0`;
        version_picker.appendChild(createOption(version, version === current_doc_version));
    }

    version_picker.style.color = "#000";
    version_picker.onchange = (event) => {
        const url_parts = window.location.pathname.split("/");
        url_parts[1] = event.target.value;
        window.location.href = url_parts.join("/");
    };

    const span = document.createElement("span");
    span.innerHTML = getHtmlForSwitcher(isOldVersion, el);
    span.appendChild(version_picker);

    el.appendChild(span);

    const rustdoc_container = document.createElement("div");

    let medium_version = current_doc_version.split(".").slice(1, 2);
    if (medium_version.length === 0) {
        medium_version = ["-1"];
    }
    medium_version = parseInt(medium_version[0]);
    if (medium_version < 0 || medium_version > 58) {
        setupStyleFor59(rustdoc_container, el);
    } else if (medium_version > 31) {
        setupStyleFor32(rustdoc_container, el, "");
    } else if (medium_version > 21) {
        setupStyleFor22(rustdoc_container, el);
    } else {
        setupStyleFor21(rustdoc_container, el);
    }

    rustdoc_container.className = document.body.className;
    document.body.className = "";
    while (document.body.childNodes.length > 0) {
        rustdoc_container.appendChild(document.body.childNodes[0]);
    }

    document.body.appendChild(el);
    document.body.appendChild(rustdoc_container);
}

showSwitcher(checkIfIsOldVersion());

}());
