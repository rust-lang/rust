var darkThemes = ["dark", "ayu"];
window.currentTheme = document.getElementById("themeStyle");
window.mainTheme = document.getElementById("mainThemeStyle");

var settingsDataset = (function () {
    var settingsElement = document.getElementById("default-settings");
    if (settingsElement === null) {
        return null;
    }
    var dataset = settingsElement.dataset;
    if (dataset === undefined) {
        return null;
    }
    return dataset;
})();

function getSettingValue(settingName) {
    var current = getCurrentValue(settingName);
    if (current !== null) {
        return current;
    }
    if (settingsDataset !== null) {
        // See the comment for `default_settings.into_iter()` etc. in
        // `Options::from_matches` in `librustdoc/config.rs`.
        var def = settingsDataset[settingName.replace(/-/g,'_')];
        if (def !== undefined) {
            return def;
        }
    }
    return null;
}

var localStoredTheme = getSettingValue("theme");

var savedHref = [];

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
        var length = arr.length;
        var i;
        if (reversed) {
            for (i = length - 1; i >= 0; --i) {
                if (func(arr[i])) {
                    return true;
                }
            }
        } else {
            for (i = 0; i < length; ++i) {
                if (func(arr[i])) {
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

// eslint-disable-next-line no-unused-vars
function hasOwnPropertyRustdoc(obj, property) {
    return Object.prototype.hasOwnProperty.call(obj, property);
}

function updateLocalStorage(name, value) {
    try {
        window.localStorage.setItem("rustdoc-" + name, value);
    } catch(e) {
        // localStorage is not accessible, do nothing
    }
}

function getCurrentValue(name) {
    try {
        return window.localStorage.getItem("rustdoc-" + name);
    } catch(e) {
        return null;
    }
}

function switchTheme(styleElem, mainStyleElem, newTheme, saveTheme) {
    var newHref = mainStyleElem.href.replace(
        /\/rustdoc([^/]*)\.css/, "/" + newTheme + "$1" + ".css");

    // If this new value comes from a system setting or from the previously
    // saved theme, no need to save it.
    if (saveTheme) {
        updateLocalStorage("theme", newTheme);
    }

    if (styleElem.href === newHref) {
        return;
    }

    var found = false;
    if (savedHref.length === 0) {
        onEachLazy(document.getElementsByTagName("link"), function(el) {
            savedHref.push(el.href);
        });
    }
    onEach(savedHref, function(el) {
        if (el === newHref) {
            found = true;
            return true;
        }
    });
    if (found) {
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
    var toggle = document.getElementById("use-system-theme");
    if (toggle && toggle instanceof HTMLInputElement) {
        toggle.checked = value;
    }
}

var updateSystemTheme = (function() {
    if (!window.matchMedia) {
        // fallback to the CSS computed value
        return function() {
            var cssTheme = getComputedStyle(document.documentElement)
                .getPropertyValue('content');

            switchTheme(
                window.currentTheme,
                window.mainTheme,
                JSON.parse(cssTheme) || "light",
                true
            );
        };
    }

    // only listen to (prefers-color-scheme: dark) because light is the default
    var mql = window.matchMedia("(prefers-color-scheme: dark)");

    function handlePreferenceChange(mql) {
        let use = function(theme) {
            switchTheme(window.currentTheme, window.mainTheme, theme, true);
        };
        // maybe the user has disabled the setting in the meantime!
        if (getSettingValue("use-system-theme") !== "false") {
            var lightTheme = getSettingValue("preferred-light-theme") || "light";
            var darkTheme = getSettingValue("preferred-dark-theme") || "dark";

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

    return function() {
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

// If we navigate away (for example to a settings page), and then use the back or
// forward button to get back to a page, the theme may have changed in the meantime.
// But scripts may not be re-loaded in such a case due to the bfcache
// (https://web.dev/bfcache/). The "pageshow" event triggers on such navigations.
// Use that opportunity to update the theme.
// We use a setTimeout with a 0 timeout here to put the change on the event queue.
// For some reason, if we try to change the theme while the `pageshow` event is
// running, it sometimes fails to take effect. The problem manifests on Chrome,
// specifically when talking to a remote website with no caching.
window.addEventListener("pageshow", function(ev) {
    if (ev.persisted) {
        setTimeout(switchToSavedTheme, 0);
    }
});
