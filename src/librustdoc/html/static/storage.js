// From rust:
/* global resourcesSuffix */
/* global allThemeNames */

var savedHref = [];

function hasClass(elem, className) {
    return elem && elem.classList && elem.classList.contains(className);
}

function addClass(elem, className) {
    if (!elem || !elem.classList) {
        return;
    }
    elem.classList.add(className);
}

function removeClass(elem, className) {
    if (!elem || !elem.classList) {
        return;
    }
    elem.classList.remove(className);
}

function onEach(arr, func, reversed) {
    if (arr && arr.length > 0 && func) {
        var length = arr.length;
        var i;
        if (reversed !== true) {
            for (i = 0; i < length; ++i) {
                if (func(arr[i]) === true) {
                    return true;
                }
            }
        } else {
            for (i = length - 1; i >= 0; --i) {
                if (func(arr[i]) === true) {
                    return true;
                }
            }
        }
    }
    return false;
}

function onEachLazy(lazyArray, func, reversed) {
    return onEach(
        Array.prototype.slice.call(lazyArray),
        func,
        reversed);
}

function hasOwnProperty(obj, property) {
    return Object.prototype.hasOwnProperty.call(obj, property);
}

function usableLocalStorage() {
    // Check if the browser supports localStorage at all:
    if (typeof Storage === "undefined") {
        return false;
    }
    // Check if we can access it; this access will fail if the browser
    // preferences deny access to localStorage, e.g., to prevent storage of
    // "cookies" (or cookie-likes, as is the case here).
    try {
        return window.localStorage !== null && window.localStorage !== undefined;
    } catch(err) {
        // Storage is supported, but browser preferences deny access to it.
        return false;
    }
}

function updateLocalStorage(name, value) {
    if (usableLocalStorage()) {
        localStorage[name] = value;
    } else {
        // No Web Storage support so we do nothing
    }
}

function getCurrentValue(name) {
    if (usableLocalStorage() && localStorage[name] !== undefined) {
        return localStorage[name];
    }
    return null;
}

function switchTheme(newTheme, saveTheme) {
    // The theme file we are switching to
    var newThemeFile = newTheme + resourcesSuffix + ".css";

    var found = false;
    if (savedHref.length === 0) {
        onEachLazy(document.getElementsByTagName("link"), function(el) {
            savedHref.push(el.href);
        });
    }
    onEach(savedHref, function(href) {
        if (href.endsWith(newThemeFile)) {
            found = true;
            return true;
        }
    });
    if (found === true) {
        onEach(allThemeNames, function(themeName) {
            // The theme file for this theme name
            var themeFile = themeName + resourcesSuffix + ".css";
            var themeSheet = document.querySelector("[href$='" + themeFile + "']");

            if (themeName === newTheme) {
                themeSheet.disabled = false;
            } else {
                themeSheet.disabled = true;
            }
        });
        // If this new value comes from a system setting or from the previously saved theme, no
        // need to save it.
        if (saveTheme === true) {
            updateLocalStorage("rustdoc-theme", newTheme);
        }
    }
}

function getSystemValue() {
    var property = getComputedStyle(document.documentElement).getPropertyValue('content');
    return property.replace(/[\"\']/g, "");
}

switchTheme(getCurrentValue("rustdoc-theme") || getSystemValue() || "light", false);
