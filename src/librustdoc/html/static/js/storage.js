// storage.js is loaded in the `<head>` of all rustdoc pages and doesn't
// use `async` or `defer`. That means it blocks further parsing and rendering
// of the page: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script.
// This makes it the correct place to act on settings that affect the display of
// the page, so we don't see major layout changes during the load of the page.
"use strict";

const darkThemes = ["dark", "ayu"];
window.currentTheme = document.getElementById("themeStyle");
window.mainTheme = document.getElementById("mainThemeStyle");

// WARNING: RUSTDOC_MOBILE_BREAKPOINT MEDIA QUERY
// If you update this line, then you also need to update the media query with the same
// warning in rustdoc.css
window.RUSTDOC_MOBILE_BREAKPOINT = 700;

const settingsDataset = (function() {
    const settingsElement = document.getElementById("default-settings");
    if (settingsElement === null) {
        return null;
    }
    const dataset = settingsElement.dataset;
    if (dataset === undefined) {
        return null;
    }
    return dataset;
})();

function getSettingValue(settingName) {
    const current = getCurrentValue(settingName);
    if (current !== null) {
        return current;
    }
    if (settingsDataset !== null) {
        // See the comment for `default_settings.into_iter()` etc. in
        // `Options::from_matches` in `librustdoc/config.rs`.
        const def = settingsDataset[settingName.replace(/-/g,"_")];
        if (def !== undefined) {
            return def;
        }
    }
    return null;
}

const localStoredTheme = getSettingValue("theme");

const savedHref = [];

// eslint-disable-next-line no-unused-vars
function hasClass(elem, className) {
    return elem && elem.classList && elem.classList.contains(className);
}

// eslint-disable-next-line no-unused-vars
function addClass(elem, className) {
    if (!elem || !elem.classList) {
        return;
    }
    elem.classList.add(className);
}

// eslint-disable-next-line no-unused-vars
function removeClass(elem, className) {
    if (!elem || !elem.classList) {
        return;
    }
    elem.classList.remove(className);
}

/**
 * Run a callback for every element of an Array.
 * @param {Array<?>}    arr        - The array to iterate over
 * @param {function(?)} func       - The callback
 * @param {boolean}     [reversed] - Whether to iterate in reverse
 */
function onEach(arr, func, reversed) {
    if (arr && arr.length > 0 && func) {
        if (reversed) {
            const length = arr.length;
            for (let i = length - 1; i >= 0; --i) {
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

function switchTheme(styleElem, mainStyleElem, newThemeName, saveTheme) {
    // If this new value comes from a system setting or from the previously
    // saved theme, no need to save it.
    if (saveTheme) {
        updateLocalStorage("theme", newThemeName);
    }

    if (savedHref.length === 0) {
        onEachLazy(document.getElementsByTagName("link"), el => {
            savedHref.push(el.href);
        });
    }
    const newHref = savedHref.find(url => {
        const m = url.match(/static\.files\/(.*)-[a-f0-9]{16}\.css$/);
        if (m && m[1] === newThemeName) {
            return true;
        }
        const m2 = url.match(/\/([^/]*)\.css$/);
        if (m2 && m2[1].startsWith(newThemeName)) {
            return true;
        }
    });
    if (newHref && newHref !== styleElem.href) {
        styleElem.href = newHref;
    }
}

// This function is called from "main.js".
// eslint-disable-next-line no-unused-vars
function useSystemTheme(value) {
    if (value === undefined) {
        value = true;
    }

    updateLocalStorage("use-system-theme", value);

    // update the toggle if we're on the settings page
    const toggle = document.getElementById("use-system-theme");
    if (toggle && toggle instanceof HTMLInputElement) {
        toggle.checked = value;
    }
}

const updateSystemTheme = (function() {
    if (!window.matchMedia) {
        // fallback to the CSS computed value
        return () => {
            const cssTheme = getComputedStyle(document.documentElement)
                .getPropertyValue("content");

            switchTheme(
                window.currentTheme,
                window.mainTheme,
                JSON.parse(cssTheme) || "light",
                true
            );
        };
    }

    // only listen to (prefers-color-scheme: dark) because light is the default
    const mql = window.matchMedia("(prefers-color-scheme: dark)");

    function handlePreferenceChange(mql) {
        const use = theme => {
            switchTheme(window.currentTheme, window.mainTheme, theme, true);
        };
        // maybe the user has disabled the setting in the meantime!
        if (getSettingValue("use-system-theme") !== "false") {
            const lightTheme = getSettingValue("preferred-light-theme") || "light";
            const darkTheme = getSettingValue("preferred-dark-theme") || "dark";

            if (mql.matches) {
                use(darkTheme);
            } else {
                // prefers a light theme, or has no preference
                use(lightTheme);
            }
            // note: we save the theme so that it doesn't suddenly change when
            // the user disables "use-system-theme" and reloads the page or
            // navigates to another page
        } else {
            use(getSettingValue("theme"));
        }
    }

    mql.addListener(handlePreferenceChange);

    return () => {
        handlePreferenceChange(mql);
    };
})();

function switchToSavedTheme() {
    switchTheme(
        window.currentTheme,
        window.mainTheme,
        getSettingValue("theme") || "light",
        false
    );
}

if (getSettingValue("use-system-theme") !== "false" && window.matchMedia) {
    // update the preferred dark theme if the user is already using a dark theme
    // See https://github.com/rust-lang/rust/pull/77809#issuecomment-707875732
    if (getSettingValue("use-system-theme") === null
        && getSettingValue("preferred-dark-theme") === null
        && darkThemes.indexOf(localStoredTheme) >= 0) {
        updateLocalStorage("preferred-dark-theme", localStoredTheme);
    }

    // call the function to initialize the theme at least once!
    updateSystemTheme();
} else {
    switchToSavedTheme();
}

if (getSettingValue("source-sidebar-show") === "true") {
    // At this point in page load, `document.body` is not available yet.
    // Set a class on the `<html>` element instead.
    addClass(document.documentElement, "source-sidebar-expanded");
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
        setTimeout(switchToSavedTheme, 0);
    }
});
