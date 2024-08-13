function storeValue(settingName, value) {
    try {
        localStorage.setItem(`clippy-lint-list-${settingName}`, value);
    } catch (e) { }
}

function loadValue(settingName) {
    return localStorage.getItem(`clippy-lint-list-${settingName}`);
}

function setTheme(theme, store) {
    let enableHighlight = false;
    let enableNight = false;
    let enableAyu = false;

    switch(theme) {
        case "ayu":
            enableAyu = true;
            break;
        case "coal":
        case "navy":
            enableNight = true;
            break;
        case "rust":
            enableHighlight = true;
            break;
        default:
            enableHighlight = true;
            theme = "light";
            break;
    }

    document.getElementsByTagName("body")[0].className = theme;

    document.getElementById("githubLightHighlight").disabled = enableNight || !enableHighlight;
    document.getElementById("githubDarkHighlight").disabled = !enableNight && !enableAyu;

    document.getElementById("styleHighlight").disabled = !enableHighlight;
    document.getElementById("styleNight").disabled = !enableNight;
    document.getElementById("styleAyu").disabled = !enableAyu;

    if (store) {
        storeValue("theme", theme);
    } else {
        document.getElementById(`theme-choice`).value = theme;
    }
}

// loading the theme after the initial load
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
const theme = loadValue('theme');
if (prefersDark.matches && !theme) {
    setTheme("coal", false);
} else {
    setTheme(theme, false);
}
let disableShortcuts = loadValue('disable-shortcuts') === "true";
document.getElementById("disable-shortcuts").checked = disableShortcuts;

window.searchState = {
    timeout: null,
    inputElem: document.getElementById("search-input"),
    lastSearch: '',
    clearInput: () => {
        searchState.inputElem.value = "";
        searchState.filterLints();
    },
    clearInputTimeout: () => {
        if (searchState.timeout !== null) {
            clearTimeout(searchState.timeout);
            searchState.timeout = null
        }
    },
    resetInputTimeout: () => {
        searchState.clearInputTimeout();
        setTimeout(searchState.filterLints, 50);
    },
    filterLints: () => {
        searchState.clearInputTimeout();

        let searchStr = searchState.inputElem.value.trim().toLowerCase();
        if (searchStr.startsWith("clippy::")) {
            searchStr = searchStr.slice(8);
        }
        if (searchState.lastSearch === searchStr) {
            return;
        }
        searchState.lastSearch = searchStr;
        const terms = searchStr.split(" ");

        onEachLazy(document.querySelectorAll("article"), lint => {
            // Search by id
            if (lint.id.indexOf(searchStr.replaceAll("-", "_")) !== -1) {
                lint.style.display = "";
                return;
            }
            // Search the description
            // The use of `for`-loops instead of `foreach` enables us to return early
            const docsLowerCase = lint.textContent.toLowerCase();
            for (index = 0; index < terms.length; index++) {
                // This is more likely and will therefore be checked first
                if (docsLowerCase.indexOf(terms[index]) !== -1) {
                    return;
                }

                if (lint.id.indexOf(terms[index]) !== -1) {
                    return;
                }

                lint.style.display = "none";
            }
        });
        if (searchStr.length > 0) {
            window.location.hash = `/${searchStr}`;
        } else {
            window.location.hash = '';
        }
    },
};

function handleInputChanged(event) {
    if (event.target !== document.activeElement) {
        return;
    }
    searchState.resetInputTimeout();
}

function handleShortcut(ev) {
    if (ev.ctrlKey || ev.altKey || ev.metaKey || disableShortcuts) {
        return;
    }

    if (document.activeElement.tagName === "INPUT") {
        if (ev.key === "Escape") {
            document.activeElement.blur();
        }
    } else {
        switch (ev.key) {
            case "s":
            case "S":
            case "/":
                ev.preventDefault(); // To prevent the key to be put into the input.
                document.getElementById("search-input").focus();
                break;
            default:
                break;
        }
    }
}

document.addEventListener("keypress", handleShortcut);
document.addEventListener("keydown", handleShortcut);

function changeSetting(elem) {
    if (elem.id === "disable-shortcuts") {
        disableShortcuts = elem.checked;
        storeValue(elem.id, elem.checked);
    }
}

function onEachLazy(lazyArray, func) {
    const arr = Array.prototype.slice.call(lazyArray);
    for (const el of arr) {
        func(el);
    }
}

function expandLintId(lintId) {
    searchState.inputElem.value = lintId;
    searchState.filterLints();

    // Expand the lint.
    const lintElem = document.getElementById(lintId);
    const isCollapsed = lintElem.classList.remove("collapsed");
    lintElem.querySelector(".label-doc-folding").innerText = "-";
}

// Show details for one lint
function openLint(event) {
    event.preventDefault();
    event.stopPropagation();
    expandLintId(event.target.getAttribute("href").slice(1));
}

function expandLint(lintId) {
    const lintElem = document.getElementById(lintId);
    const isCollapsed = lintElem.classList.toggle("collapsed");
    lintElem.querySelector(".label-doc-folding").innerText = isCollapsed ? "+" : "-";
}

function copyToClipboard(event) {
    event.preventDefault();
    event.stopPropagation();

    const clipboard = event.target;

    let resetClipboardTimeout = null;
    const resetClipboardIcon = clipboard.innerHTML;

    function resetClipboard() {
        resetClipboardTimeout = null;
        clipboard.innerHTML = resetClipboardIcon;
    }

    navigator.clipboard.writeText("clippy::" + clipboard.parentElement.id.slice(5));

    clipboard.innerHTML = "&#10003;";
    if (resetClipboardTimeout !== null) {
        clearTimeout(resetClipboardTimeout);
    }
    resetClipboardTimeout = setTimeout(resetClipboard, 1000);
}

function handleBlur(event) {
    const parent = document.getElementById("settings-dropdown");
    if (!parent.contains(document.activeElement) &&
        !parent.contains(event.relatedTarget)
    ) {
        parent.classList.remove("open");
    }
}

function toggleExpansion(expand) {
    onEachLazy(
        document.querySelectorAll("article"),
        expand ? el => el.classList.remove("collapsed") : el => el.classList.add("collapsed"),
    );
}

function generateListOfOptions(list, elementId) {
    let html = '';
    let nbEnabled = 0;
    for (const [key, value] of Object.entries(list)) {
        const attr = value ? " checked" : "";
        html += `\
<li class="checkbox">\
    <label class="text-capitalize">\
        <input type="checkbox"${attr}/>${key}\
    </label>\
</li>`;
        if (value) {
            nbEnabled += 1;
        }
    }

    const elem = document.getElementById(elementId);
    elem.previousElementSibling.querySelector(".badge").innerText = `${nbEnabled}`;
    elem.innerHTML += html;
}

function generateSettings() {
    const settings = document.getElementById("settings-dropdown");
    const settingsButton = settings.querySelector(".settings-icon")
    settingsButton.onclick = () => settings.classList.toggle("open");
    settingsButton.onblur = handleBlur;
    const settingsMenu = settings.querySelector(".settings-menu");
    settingsMenu.onblur = handleBlur;
    onEachLazy(
        settingsMenu.querySelectorAll("input"),
        el => el.onblur = handleBlur,
    );

    const LEVEL_FILTERS_DEFAULT = {allow: true, warn: true, deny: true, none: true};
    generateListOfOptions(LEVEL_FILTERS_DEFAULT, "lint-levels");

    // Generate lint groups.
    const GROUPS_FILTER_DEFAULT = {
        cargo: true,
        complexity: true,
        correctness: true,
        deprecated: false,
        nursery: true,
        pedantic: true,
        perf: true,
        restriction: true,
        style: true,
        suspicious: true,
    };
    generateListOfOptions(GROUPS_FILTER_DEFAULT, "lint-groups");

    const APPLICABILITIES_FILTER_DEFAULT = {
        Unspecified: true,
        Unresolved: true,
        MachineApplicable: true,
        MaybeIncorrect: true,
        HasPlaceholders: true
    };
    generateListOfOptions(APPLICABILITIES_FILTER_DEFAULT, "lint-applicabilities");

    const VERSIONS_FILTERS = {
        "≥": {enabled: false, minorVersion: null },
        "≤": {enabled: false, minorVersion: null },
        "=": {enabled: false, minorVersion: null },
    };

    let html = '';
    for (const kind of ["≥", "≤", "="]) {
        html += `\
<li class="checkbox">\
    <label>${kind}</label>\
    <span>1.</span> \
    <input type="number" \
           min="29" \
           id="filter-${kind}" \
           class="version-filter-input form-control filter-input" \
           maxlength="2" \
           onchange="updateVersionFilters()" />\
    <span>.0</span>\
</li>`;
    }
    document.getElementById("version-filter-selector").innerHTML += html;
}

function generateSearch() {
    searchState.inputElem.addEventListener("change", handleInputChanged);
    searchState.inputElem.addEventListener("input", handleInputChanged);
    searchState.inputElem.addEventListener("keydown", handleInputChanged);
    searchState.inputElem.addEventListener("keyup", handleInputChanged);
    searchState.inputElem.addEventListener("paste", handleInputChanged);
}

generateSettings();
generateSearch();

function scrollToLint(lintId) {
    const target = document.getElementById(lintId);
    if (!target) {
        return;
    }
    target.scrollIntoView();
    expandLintId(lintId);
}

// If the page we arrive on has link to a given lint, we scroll to it.
function scrollToLintByURL() {
    const lintId = window.location.hash.substring(2);
    if (lintId.length > 0) {
        scrollToLint(lintId);
    }
}

scrollToLintByURL();

onEachLazy(document.querySelectorAll("pre > code.language-rust"), el => hljs.highlightElement(el));
