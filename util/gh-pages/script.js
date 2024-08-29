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
        function matchesSearch(lint, terms, searchStr) {
            // Search by id
            if (lint.elem.id.indexOf(searchStr) !== -1) {
                return true;
            }
            // Search the description
            // The use of `for`-loops instead of `foreach` enables us to return early
            const docsLowerCase = lint.elem.textContent.toLowerCase();
            for (const term of terms) {
                // This is more likely and will therefore be checked first
                if (docsLowerCase.indexOf(term) !== -1) {
                    return true;
                }

                if (lint.elem.id.indexOf(term) !== -1) {
                    return true;
                }

                return false;
            }
            return true;
        }

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
        const cleanedSearchStr = searchStr.replaceAll("-", "_");

        for (const lint of filters.getAllLints()) {
            lint.searchFilteredOut = !matchesSearch(lint, terms, cleanedSearchStr);
            if (lint.filteredOut) {
                continue;
            }
            if (lint.searchFilteredOut) {
                lint.elem.style.display = "none";
            } else {
                lint.elem.style.display = "";
            }
        }
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

function toggleElements(filter, value) {
    let needsUpdate = false;
    let count = 0;

    const element = document.getElementById(filters[filter].id);
    onEachLazy(
        element.querySelectorAll("ul input"),
        el => {
            if (el.checked !== value) {
                el.checked = value;
                filters[filter][el.getAttribute("data-value")] = value;
                needsUpdate = true;
            }
            count += 1;
        }
    );
    element.querySelector(".badge").innerText = value ? count : 0;
    if (needsUpdate) {
        filters.filterLints();
    }
}

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

function highlightIfNeeded(elem) {
    onEachLazy(elem.querySelectorAll("pre > code.language-rust:not(.highlighted)"), el => {
        hljs.highlightElement(el.parentElement)
        el.classList.add("highlighted");
    });
}

function expandLint(lintId) {
    const lintElem = document.getElementById(lintId);
    const isCollapsed = lintElem.classList.toggle("collapsed");
    lintElem.querySelector(".label-doc-folding").innerText = isCollapsed ? "+" : "−";
    highlightIfNeeded(lintElem);
}

// Show details for one lint
function openLint(event) {
    event.preventDefault();
    event.stopPropagation();
    expandLint(event.target.getAttribute("href").slice(1));
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

function handleBlur(event, elementId) {
    const parent = document.getElementById(elementId);
    if (!parent.contains(document.activeElement) &&
        !parent.contains(event.relatedTarget)
    ) {
        parent.classList.remove("open");
    }
}

function toggleExpansion(expand) {
    onEachLazy(
        document.querySelectorAll("article"),
        expand ? el => {
            el.classList.remove("collapsed");
            highlightIfNeeded(el);
        } : el => el.classList.add("collapsed"),
    );
}

// Returns the current URL without any query parameter or hash.
function getNakedUrl() {
    return window.location.href.split("?")[0].split("#")[0];
}

const GROUPS_FILTER_DEFAULT = {
    cargo: true,
    complexity: true,
    correctness: true,
    nursery: true,
    pedantic: true,
    perf: true,
    restriction: true,
    style: true,
    suspicious: true,
    deprecated: false,
};
const LEVEL_FILTERS_DEFAULT = {
    allow: true,
    warn: true,
    deny: true,
    none: true,
};
const APPLICABILITIES_FILTER_DEFAULT = {
    Unspecified: true,
    MachineApplicable: true,
    MaybeIncorrect: true,
    HasPlaceholders: true,
};
const URL_PARAMS_CORRESPONDANCE = {
    "groups_filter": "groups",
    "levels_filter": "levels",
    "applicabilities_filter": "applicabilities",
    "version_filter": "versions",
};
const VERSIONS_CORRESPONDANCE = {
    "lte": "≤",
    "gte": "≥",
    "eq": "=",
};

window.filters = {
    groups_filter: { id: "lint-groups", ...GROUPS_FILTER_DEFAULT },
    levels_filter: { id: "lint-levels", ...LEVEL_FILTERS_DEFAULT },
    applicabilities_filter: { id: "lint-applicabilities", ...APPLICABILITIES_FILTER_DEFAULT },
    version_filter: {
        "≥": null,
        "≤": null,
        "=": null,
    },
    allLints: null,
    getAllLints: () => {
        if (filters.allLints === null) {
            filters.allLints = Array.prototype.slice.call(
                document.getElementsByTagName("article"),
            ).map(elem => {
                let version = elem.querySelector(".label-version").innerText;
                // Strip the "pre " prefix for pre 1.29.0 lints
                if (version.startsWith("pre ")) {
                    version = version.slice(4);
                }
                return {
                    elem: elem,
                    group: elem.querySelector(".label-lint-group").innerText,
                    level: elem.querySelector(".label-lint-level").innerText,
                    version: parseInt(version.split(".")[1]),
                    applicability: elem.querySelector(".label-applicability").innerText,
                    filteredOut: false,
                    searchFilteredOut: false,
                };
            });
        }
        return filters.allLints;
    },
    regenerateURLparams: () => {
        const urlParams = new URLSearchParams(window.location.search);

        function compareObjects(obj1, obj2) {
            return (JSON.stringify(obj1) === JSON.stringify({ id: obj1.id, ...obj2 }));
        }
        function updateIfNeeded(filterName, obj2) {
            const obj1 = filters[filterName];
            const name = URL_PARAMS_CORRESPONDANCE[filterName];
            if (!compareObjects(obj1, obj2)) {
                urlParams.set(
                    name,
                    Object.entries(obj1).filter(
                        ([key, value]) => value && key !== "id"
                    ).map(
                        ([key, _]) => key
                    ).join(","),
                );
            } else {
                urlParams.delete(name);
            }
        }

        updateIfNeeded("groups_filter", GROUPS_FILTER_DEFAULT);
        updateIfNeeded("levels_filter", LEVEL_FILTERS_DEFAULT);
        updateIfNeeded(
            "applicabilities_filter", APPLICABILITIES_FILTER_DEFAULT);

        const versions = [];
        if (filters.version_filter["="] !== null) {
            versions.push(`eq:${filters.version_filter["="]}`);
        }
        if (filters.version_filter["≥"] !== null) {
            versions.push(`gte:${filters.version_filter["≥"]}`);
        }
        if (filters.version_filter["≤"] !== null) {
            versions.push(`lte:${filters.version_filter["≤"]}`);
        }
        if (versions.length !== 0) {
            urlParams.set(URL_PARAMS_CORRESPONDANCE["version_filter"], versions.join(","));
        } else {
            urlParams.delete(URL_PARAMS_CORRESPONDANCE["version_filter"]);
        }

        let params = urlParams.toString();
        if (params.length !== 0) {
            params = `?${params}`;
        }

        const url = getNakedUrl() + params + window.location.hash
        if (!history.state) {
            history.pushState(null, "", url);
        } else {
            history.replaceState(null, "", url);
        }
    },
    filterLints: () => {
        // First we regenerate the URL parameters.
        filters.regenerateURLparams();
        for (const lint of filters.getAllLints()) {
            lint.filteredOut = (!filters.groups_filter[lint.group]
                || !filters.levels_filter[lint.level]
                || !filters.applicabilities_filter[lint.applicability]
                || !(filters.version_filter["="] === null || lint.version === filters.version_filter["="])
                || !(filters.version_filter["≥"] === null || lint.version > filters.version_filter["≥"])
                || !(filters.version_filter["≤"] === null || lint.version < filters.version_filter["≤"])
            );
            if (lint.filteredOut || lint.searchFilteredOut) {
                lint.elem.style.display = "none";
            } else {
                lint.elem.style.display = "";
            }
        }
    },
};

function updateFilter(elem, filter, skipLintsFiltering) {
    const value = elem.getAttribute("data-value");
    if (filters[filter][value] !== elem.checked) {
        filters[filter][value] = elem.checked;
        const counter = document.querySelector(`#${filters[filter].id} .badge`);
        counter.innerText = parseInt(counter.innerText) + (elem.checked ? 1 : -1);
        if (!skipLintsFiltering) {
            filters.filterLints();
        }
    }
}

function updateVersionFilters(elem, skipLintsFiltering) {
    let value = elem.value.trim();
    if (value.length === 0) {
        value = null;
    } else if (/^\d+$/.test(value)) {
        value = parseInt(value);
    } else {
        console.error(`Failed to get version number from "${value}"`);
        return;
    }

    const counter = document.querySelector("#version-filter .badge");
    let count = 0;
    onEachLazy(document.querySelectorAll("#version-filter input"), el => {
        if (el.value.trim().length !== 0) {
            count += 1;
        }
    });
    counter.innerText = count;

    const comparisonKind = elem.getAttribute("data-value");
    if (filters.version_filter[comparisonKind] !== value) {
        filters.version_filter[comparisonKind] = value;
        if (!skipLintsFiltering) {
            filters.filterLints();
        }
    }
}

function clearVersionFilters() {
    let needsUpdate = false;

    onEachLazy(document.querySelectorAll("#version-filter input"), el => {
        el.value = "";
        const comparisonKind = el.getAttribute("data-value");
        if (filters.version_filter[comparisonKind] !== null) {
            needsUpdate = true;
            filters.version_filter[comparisonKind] = null;
        }
    });
    document.querySelector("#version-filter .badge").innerText = 0;
    if (needsUpdate) {
        filters.filterLints();
    }
}

function resetGroupsToDefault() {
    let needsUpdate = false;
    let count = 0;

    onEachLazy(document.querySelectorAll("#lint-groups-selector input"), el => {
        const key = el.getAttribute("data-value");
        const value = GROUPS_FILTER_DEFAULT[key];
        if (filters.groups_filter[key] !== value) {
            filters.groups_filter[key] = value;
            el.checked = value;
            needsUpdate = true;
        }
        if (value) {
            count += 1;
        }
    });
    document.querySelector("#lint-groups .badge").innerText = count;
    if (needsUpdate) {
        filters.filterLints();
    }
}

function generateListOfOptions(list, elementId, filter) {
    let html = '';
    let nbEnabled = 0;
    for (const [key, value] of Object.entries(list)) {
        const attr = value ? " checked" : "";
        html += `\
<li class="checkbox">\
    <label class="text-capitalize">\
        <input type="checkbox" data-value="${key}" \
               onchange="updateFilter(this, '${filter}')"${attr}/>${key}\
    </label>\
</li>`;
        if (value) {
            nbEnabled += 1;
        }
    }

    const elem = document.getElementById(`${elementId}-selector`);
    elem.previousElementSibling.querySelector(".badge").innerText = `${nbEnabled}`;
    elem.innerHTML += html;

    setupDropdown(elementId);
}

function setupDropdown(elementId) {
    const elem = document.getElementById(elementId);
    const button = document.querySelector(`#${elementId} > button`);
    button.onclick = () => elem.classList.toggle("open");

    const setBlur = child => {
        child.onblur = event => handleBlur(event, elementId);
    };
    onEachLazy(elem.children, setBlur);
    onEachLazy(elem.querySelectorAll("select"), setBlur);
    onEachLazy(elem.querySelectorAll("input"), setBlur);
    onEachLazy(elem.querySelectorAll("ul button"), setBlur);
}

function generateSettings() {
    setupDropdown("settings-dropdown");

    generateListOfOptions(LEVEL_FILTERS_DEFAULT, "lint-levels", "levels_filter");
    generateListOfOptions(GROUPS_FILTER_DEFAULT, "lint-groups", "groups_filter");
    generateListOfOptions(
        APPLICABILITIES_FILTER_DEFAULT, "lint-applicabilities", "applicabilities_filter");

    let html = '';
    for (const kind of ["≥", "≤", "="]) {
        html += `\
<li class="checkbox">\
    <label>${kind}</label>\
    <span>1.</span> \
    <input type="number" \
           min="29" \
           class="version-filter-input form-control filter-input" \
           maxlength="2" \
           data-value="${kind}" \
           onchange="updateVersionFilters(this)" \
           oninput="updateVersionFilters(this)" \
           onkeydown="updateVersionFilters(this)" \
           onkeyup="updateVersionFilters(this)" \
           onpaste="updateVersionFilters(this)" \
    />
    <span>.0</span>\
</li>`;
    }
    document.getElementById("version-filter-selector").innerHTML += html;
    setupDropdown("version-filter");
}

function generateSearch() {
    searchState.inputElem.addEventListener("change", handleInputChanged);
    searchState.inputElem.addEventListener("input", handleInputChanged);
    searchState.inputElem.addEventListener("keydown", handleInputChanged);
    searchState.inputElem.addEventListener("keyup", handleInputChanged);
    searchState.inputElem.addEventListener("paste", handleInputChanged);
}

function scrollToLint(lintId) {
    const target = document.getElementById(lintId);
    if (!target) {
        return;
    }
    target.scrollIntoView();
    expandLint(lintId);
}

// If the page we arrive on has link to a given lint, we scroll to it.
function scrollToLintByURL() {
    const lintId = window.location.hash.substring(2);
    if (lintId.length > 0) {
        scrollToLint(lintId);
    }
}

function parseURLFilters() {
    const urlParams = new URLSearchParams(window.location.search);

    for (const [key, value] of urlParams.entries()) {
        for (const [corres_key, corres_value] of Object.entries(URL_PARAMS_CORRESPONDANCE)) {
            if (corres_value === key) {
                if (key !== "versions") {
                    const settings  = new Set(value.split(","));
                    onEachLazy(document.querySelectorAll(`#lint-${key} ul input`), elem => {
                        elem.checked = settings.has(elem.getAttribute("data-value"));
                        updateFilter(elem, corres_key, true);
                    });
                } else {
                    const settings = value.split(",").map(elem => elem.split(":"));

                    for (const [kind, value] of settings) {
                        const elem = document.querySelector(
                            `#version-filter input[data-value="${VERSIONS_CORRESPONDANCE[kind]}"]`);
                        elem.value = value;
                        updateVersionFilters(elem, true);
                    }
                }
            }
        }
    }
}

document.getElementById(`theme-choice`).value = loadValue("theme");
let disableShortcuts = loadValue('disable-shortcuts') === "true";
document.getElementById("disable-shortcuts").checked = disableShortcuts;

document.addEventListener("keypress", handleShortcut);
document.addEventListener("keydown", handleShortcut);

generateSettings();
generateSearch();
parseURLFilters();
scrollToLintByURL();
filters.filterLints();
