// storage.js is loaded in the `<head>` of all rustdoc pages and doesn't
// use `async` or `defer`. That means it blocks further parsing and rendering
// of the page: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script.
// This makes it the correct place to act on settings that affect the display of
// the page, so we don't see major layout changes during the load of the page.
"use strict";

const darkThemes = ["dark", "ayu"];
window.currentTheme = document.getElementById("themeStyle");

const settingsDataset = (function() {
    const settingsElement = document.getElementById("default-settings");
    return settingsElement && settingsElement.dataset ? settingsElement.dataset : null;
})();

function getSettingValue(settingName) {
    const current = getCurrentValue(settingName);
    if (current === null && settingsDataset !== null) {
        // See the comment for `default_settings.into_iter()` etc. in
        // `Options::from_matches` in `librustdoc/config.rs`.
        const def = settingsDataset[settingName.replace(/-/g,"_")];
        if (def !== undefined) {
            return def;
        }
    }
    return current;
}

const localStoredTheme = getSettingValue("theme");

// eslint-disable-next-line no-unused-vars
function hasClass(elem, className) {
    return elem && elem.classList && elem.classList.contains(className);
}

function addClass(elem, className) {
    if (elem && elem.classList) {
        elem.classList.add(className);
    }
}

// eslint-disable-next-line no-unused-vars
function removeClass(elem, className) {
    if (elem && elem.classList) {
        elem.classList.remove(className);
    }
}

/**
 * Run a callback for every element of an Array.
 * @param {Array<?>}    arr        - The array to iterate over
 * @param {function(?)} func       - The callback
 * @param {boolean}     [reversed] - Whether to iterate in reverse
 */
function onEach(arr, func, reversed) {
    if (arr && arr.length > 0) {
        if (reversed) {
            for (let i = arr.length - 1; i >= 0; --i) {
                if (func(arr[i])) {
                    return true;
                }
            }
        } else {
            for (const elem of arr) {
                if (func(elem)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * Turn an HTMLCollection or a NodeList into an Array, then run a callback
 * for every element. This is useful because iterating over an HTMLCollection
 * or a "live" NodeList while modifying it can be very slow.
 * https://developer.mozilla.org/en-US/docs/Web/API/HTMLCollection
 * https://developer.mozilla.org/en-US/docs/Web/API/NodeList
 * @param {NodeList<?>|HTMLCollection<?>} lazyArray  - An array to iterate over
 * @param {function(?)}                   func       - The callback
 * @param {boolean}                       [reversed] - Whether to iterate in reverse
 */
// eslint-disable-next-line no-unused-vars
function onEachLazy(lazyArray, func, reversed) {
    return onEach(
        Array.prototype.slice.call(lazyArray),
        func,
        reversed);
}

function updateLocalStorage(name, value) {
    try {
        window.localStorage.setItem("rustdoc-" + name, value);
    } catch (e) {
        // localStorage is not accessible, do nothing
    }
}

function getCurrentValue(name) {
    try {
        return window.localStorage.getItem("rustdoc-" + name);
    } catch (e) {
        return null;
    }
}

// Get a value from the rustdoc-vars div, which is used to convey data from
// Rust to the JS. If there is no such element, return null.
const getVar = (function getVar(name) {
    const el = document.querySelector("head > meta[name='rustdoc-vars']");
    return el ? el.attributes["data-" + name].value : null;
});

function switchTheme(newThemeName, saveTheme) {
    // If this new value comes from a system setting or from the previously
    // saved theme, no need to save it.
    if (saveTheme) {
        updateLocalStorage("theme", newThemeName);
    }

    let newHref;

    if (newThemeName === "light" || newThemeName === "dark" || newThemeName === "ayu") {
        newHref = getVar("static-root-path") + getVar("theme-" + newThemeName + "-css");
    } else {
        newHref = getVar("root-path") + newThemeName + getVar("resource-suffix") + ".css";
    }

    if (!window.currentTheme) {
        document.write(`<link rel="stylesheet" id="themeStyle" href="${newHref}">`);
        window.currentTheme = document.getElementById("themeStyle");
    } else if (newHref !== window.currentTheme.href) {
        window.currentTheme.href = newHref;
    }
}

const updateTheme = (function() {
    // only listen to (prefers-color-scheme: dark) because light is the default
    const mql = window.matchMedia("(prefers-color-scheme: dark)");

    /**
     * Update the current theme to match whatever the current combination of
     * * the preference for using the system theme
     *   (if this is the case, the value of preferred-light-theme, if the
     *   system theme is light, otherwise if dark, the value of
     *   preferred-dark-theme.)
     * * the preferred theme
     * … dictates that it should be.
     */
    function updateTheme() {
        // maybe the user has disabled the setting in the meantime!
        if (getSettingValue("use-system-theme") !== "false") {
            const lightTheme = getSettingValue("preferred-light-theme") || "light";
            const darkTheme = getSettingValue("preferred-dark-theme") || "dark";
            updateLocalStorage("use-system-theme", "true");

            // use light theme if user prefers it, or has no preference
            switchTheme(mql.matches ? darkTheme : lightTheme, true);
            // note: we save the theme so that it doesn't suddenly change when
            // the user disables "use-system-theme" and reloads the page or
            // navigates to another page
        } else {
            switchTheme(getSettingValue("theme"), false);
        }
    }

    mql.addEventListener("change", updateTheme);

    return updateTheme;
})();

if (getSettingValue("use-system-theme") !== "false" && window.matchMedia) {
    // update the preferred dark theme if the user is already using a dark theme
    // See https://github.com/rust-lang/rust/pull/77809#issuecomment-707875732
    if (getSettingValue("use-system-theme") === null
        && getSettingValue("preferred-dark-theme") === null
        && darkThemes.indexOf(localStoredTheme) >= 0) {
        updateLocalStorage("preferred-dark-theme", localStoredTheme);
    }
}

updateTheme();

if (getSettingValue("source-sidebar-show") === "true") {
    // At this point in page load, `document.body` is not available yet.
    // Set a class on the `<html>` element instead.
    addClass(document.documentElement, "src-sidebar-expanded");
}

// If we navigate away (for example to a settings page), and then use the back or
// forward button to get back to a page, the theme may have changed in the meantime.
// But scripts may not be re-loaded in such a case due to the bfcache
// (https://web.dev/bfcache/). The "pageshow" event triggers on such navigations.
// Use that opportunity to update the theme.
// We use a setTimeout with a 0 timeout here to put the change on the event queue.
// For some reason, if we try to change the theme while the `pageshow` event is
// running, it sometimes fails to take effect. The problem manifests on Chrome,
// specifically when talking to a remote website with no caching.
window.addEventListener("pageshow", ev => {
    if (ev.persisted) {
        setTimeout(updateTheme, 0);
    }
});
