// storage.js is loaded in the `<head>` of all rustdoc pages and doesn't
// use `async` or `defer`. That means it blocks further parsing and rendering
// of the page: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script.
// This makes it the correct place to act on settings that affect the display of
// the page, so we don't see major layout changes during the load of the page.
"use strict";

/**
 * @import * as rustdoc from "./rustdoc.d.ts";
 */

const builtinThemes = ["light", "dark", "ayu"];
const darkThemes = ["dark", "ayu"];
window.currentTheme = (function() {
    const currentTheme = document.getElementById("themeStyle");
    return currentTheme instanceof HTMLLinkElement ? currentTheme : null;
})();

const settingsDataset = (function() {
    const settingsElement = document.getElementById("default-settings");
    return settingsElement && settingsElement.dataset ? settingsElement.dataset : null;
})();

/**
 * Assert that the passed value is nonnull, then return it.
 *
 * Takes an optional error message argument.
 *
 * Must be defined in this file, as it is loaded before all others.
 *
 * @template T
 * @param {T|null} x
 * @param {string=} msg
 * @returns T
 */
// used in other files, not yet used in this one.
// eslint-disable-next-line no-unused-vars
function nonnull(x, msg) {
    if (x === null) {
        throw (msg || "unexpected null value!");
    } else {
        return x;
    }
}

/**
 * Assert that the passed value is not undefined, then return it.
 *
 * Takes an optional error message argument.
 *
 * Must be defined in this file, as it is loaded before all others.
 *
 * @template T
 * @param {T|undefined} x
 * @param {string=} msg
 * @returns T
 */
// used in other files, not yet used in this one.
// eslint-disable-next-line no-unused-vars
function nonundef(x, msg) {
    if (x === undefined) {
        throw (msg || "unexpected null value!");
    } else {
        return x;
    }
}

/**
 * Get a configuration value. If it's not set, get the default.
 *
 * @param {string} settingName
 * @returns
 */
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

/**
 * Check if a DOM Element has the given class set.
 * If `elem` is null, returns false.
 *
 * @param {HTMLElement|null} elem
 * @param {string} className
 * @returns {boolean}
 */
// eslint-disable-next-line no-unused-vars
function hasClass(elem, className) {
    return !!elem && !!elem.classList && elem.classList.contains(className);
}

/**
 * Add a class to a DOM Element. If `elem` is null,
 * does nothing. This function is idempotent.
 *
 * @param {Element|null} elem
 * @param {string} className
 */
function addClass(elem, className) {
    if (elem && elem.classList) {
        elem.classList.add(className);
    }
}

/**
 * Remove a class from a DOM Element. If `elem` is null,
 * does nothing. This function is idempotent.
 *
 * @param {Element|null} elem
 * @param {string} className
 */
// eslint-disable-next-line no-unused-vars
function removeClass(elem, className) {
    if (elem && elem.classList) {
        elem.classList.remove(className);
    }
}

/**
 * Run a callback for every element of an Array.
 * @param {Array<?>}                       arr  - The array to iterate over
 * @param {function(?): boolean|void} func - The callback
 */
function onEach(arr, func) {
    for (const elem of arr) {
        if (func(elem)) {
            return true;
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
 * @param {NodeList|HTMLCollection} lazyArray  - An array to iterate over
 * @param {function(?): boolean|void}    func       - The callback
 */
// eslint-disable-next-line no-unused-vars
function onEachLazy(lazyArray, func) {
    return onEach(
        Array.prototype.slice.call(lazyArray),
        func);
}

/**
 * Set a configuration value. This uses localstorage,
 * with a `rustdoc-` prefix, to avoid clashing with other
 * web apps that may be running in the same domain (for example, mdBook).
 * If localStorage is disabled, this function does nothing.
 *
 * @param {string} name
 * @param {string|null} value
 */
function updateLocalStorage(name, value) {
    try {
        if (value === null) {
            window.localStorage.removeItem("rustdoc-" + name);
        } else {
            window.localStorage.setItem("rustdoc-" + name, value);
        }
    } catch (e) {
        // localStorage is not accessible, do nothing
    }
}

/**
 * Get a configuration value. If localStorage is disabled,
 * this function returns null. If the setting was never
 * changed by the user, it also returns null; if you want to
 * be able to use a default value, call `getSettingValue` instead.
 *
 * @param {string} name
 * @returns {string|null}
 */
function getCurrentValue(name) {
    try {
        return window.localStorage.getItem("rustdoc-" + name);
    } catch (e) {
        return null;
    }
}

/**
 * Get a value from the rustdoc-vars div, which is used to convey data from
 * Rust to the JS. If there is no such element, return null.
 *
 * @param {string} name
 * @returns {string|null}
 */
function getVar(name) {
    const el = document.querySelector("head > meta[name='rustdoc-vars']");
    return el ? el.getAttribute("data-" + name) : null;
}

/**
 * Change the current theme.
 * @param {string|null} newThemeName
 * @param {boolean} saveTheme
 */
function switchTheme(newThemeName, saveTheme) {
    const themeNames = (getVar("themes") || "").split(",").filter(t => t);
    themeNames.push(...builtinThemes);

    // Ensure that the new theme name is among the defined themes
    if (newThemeName === null || themeNames.indexOf(newThemeName) === -1) {
        return;
    }

    // If this new value comes from a system setting or from the previously
    // saved theme, no need to save it.
    if (saveTheme) {
        updateLocalStorage("theme", newThemeName);
    }

    document.documentElement.setAttribute("data-theme", newThemeName);

    if (builtinThemes.indexOf(newThemeName) !== -1) {
        if (window.currentTheme && window.currentTheme.parentNode) {
            window.currentTheme.parentNode.removeChild(window.currentTheme);
            window.currentTheme = null;
        }
    } else {
        const newHref = getVar("root-path") + encodeURIComponent(newThemeName) +
            getVar("resource-suffix") + ".css";
        if (!window.currentTheme) {
            // If we're in the middle of loading, document.write blocks
            // rendering, but if we are done, it would blank the page.
            if (document.readyState === "loading") {
                document.write(`<link rel="stylesheet" id="themeStyle" href="${newHref}">`);
                window.currentTheme = (function() {
                    const currentTheme = document.getElementById("themeStyle");
                    return currentTheme instanceof HTMLLinkElement ? currentTheme : null;
                })();
            } else {
                window.currentTheme = document.createElement("link");
                window.currentTheme.rel = "stylesheet";
                window.currentTheme.id = "themeStyle";
                window.currentTheme.href = newHref;
                document.documentElement.appendChild(window.currentTheme);
            }
        } else if (newHref !== window.currentTheme.href) {
            window.currentTheme.href = newHref;
        }
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

// @ts-ignore
if (getSettingValue("use-system-theme") !== "false" && window.matchMedia) {
    // update the preferred dark theme if the user is already using a dark theme
    // See https://github.com/rust-lang/rust/pull/77809#issuecomment-707875732
    if (getSettingValue("use-system-theme") === null
        && getSettingValue("preferred-dark-theme") === null
        && localStoredTheme !== null
        && darkThemes.indexOf(localStoredTheme) >= 0) {
        updateLocalStorage("preferred-dark-theme", localStoredTheme);
    }
}

updateTheme();

// Hide, show, and resize the sidebar at page load time
//
// This needs to be done here because this JS is render-blocking,
// so that the sidebar doesn't "jump" after appearing on screen.
// The user interaction to change this is set up in main.js.
//
// At this point in page load, `document.body` is not available yet.
// Set a class on the `<html>` element instead.
if (getSettingValue("source-sidebar-show") === "true") {
    addClass(document.documentElement, "src-sidebar-expanded");
}
if (getSettingValue("hide-sidebar") === "true") {
    addClass(document.documentElement, "hide-sidebar");
}
if (getSettingValue("hide-toc") === "true") {
    addClass(document.documentElement, "hide-toc");
}
if (getSettingValue("hide-modnav") === "true") {
    addClass(document.documentElement, "hide-modnav");
}
if (getSettingValue("sans-serif-fonts") === "true") {
    addClass(document.documentElement, "sans-serif");
}
if (getSettingValue("word-wrap-source-code") === "true") {
    addClass(document.documentElement, "word-wrap-source-code");
}
function updateSidebarWidth() {
    const desktopSidebarWidth = getSettingValue("desktop-sidebar-width");
    if (desktopSidebarWidth && desktopSidebarWidth !== "null") {
        document.documentElement.style.setProperty(
            "--desktop-sidebar-width",
            desktopSidebarWidth + "px",
        );
    }
    const srcSidebarWidth = getSettingValue("src-sidebar-width");
    if (srcSidebarWidth && srcSidebarWidth !== "null") {
        document.documentElement.style.setProperty(
            "--src-sidebar-width",
            srcSidebarWidth + "px",
        );
    }
}
updateSidebarWidth();

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
        setTimeout(updateSidebarWidth, 0);
    }
});

// Custom elements are used to insert some JS-dependent features into Rustdoc,
// because the [parser] runs the connected callback
// synchronously. It needs to be added synchronously so that nothing below it
// becomes visible until after it's done. Otherwise, you get layout jank.
//
// That's also why this is in storage.js and not main.js.
//
// [parser]: https://html.spec.whatwg.org/multipage/parsing.html
class RustdocSearchElement extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        const rootPath = getVar("root-path");
        const currentCrate = getVar("current-crate");
        this.innerHTML = `<nav class="sub">
            <form class="search-form">
                <span></span> <!-- This empty span is a hacky fix for Safari - See #93184 -->
                <div id="sidebar-button" tabindex="-1">
                    <a href="${rootPath}${currentCrate}/all.html" title="show sidebar"></a>
                </div>
                <input
                    class="search-input"
                    name="search"
                    aria-label="Run search in the documentation"
                    autocomplete="off"
                    spellcheck="false"
                    placeholder="Type ‘S’ or ‘/’ to search, ‘?’ for more options…"
                    type="search">
            </form>
        </nav>`;
    }
}
window.customElements.define("rustdoc-search", RustdocSearchElement);
class RustdocToolbarElement extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        // Avoid replacing the children after they're already here.
        if (this.firstElementChild) {
            return;
        }
        const rootPath = getVar("root-path");
        this.innerHTML = `
        <div id="settings-menu" tabindex="-1">
            <a href="${rootPath}settings.html"><span class="label">Settings</span></a>
        </div>
        <div id="help-button" tabindex="-1">
            <a href="${rootPath}help.html"><span class="label">Help</span></a>
        </div>
        <button id="toggle-all-docs"><span class="label">Summary</span></button>`;
    }
}
window.customElements.define("rustdoc-toolbar", RustdocToolbarElement);
