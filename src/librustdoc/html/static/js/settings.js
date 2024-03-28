// Local js definitions:
/* global getSettingValue, updateLocalStorage, updateTheme */
/* global addClass, removeClass, onEach, onEachLazy, blurHandler */
/* global MAIN_ID, getVar, getSettingsButton */

"use strict";

(function() {
    const isSettingsPage = window.location.pathname.endsWith("/settings.html");

    function changeSetting(settingName, value) {
        if (settingName === "theme") {
            const useSystem = value === "system preference" ? "true" : "false";
            updateLocalStorage("use-system-theme", useSystem);
        }
        updateLocalStorage(settingName, value);

        switch (settingName) {
            case "theme":
            case "preferred-dark-theme":
            case "preferred-light-theme":
                updateTheme();
                updateLightAndDark();
                break;
            case "line-numbers":
                if (value === true) {
                    window.rustdoc_add_line_numbers_to_examples();
                } else {
                    window.rustdoc_remove_line_numbers_from_examples();
                }
                break;
            case "hide-sidebar":
                if (value === true) {
                    addClass(document.documentElement, "hide-sidebar");
                } else {
                    removeClass(document.documentElement, "hide-sidebar");
                }
                break;
            case "hide-toc":
                if (value === true) {
                    addClass(document.documentElement, "hide-toc");
                } else {
                    removeClass(document.documentElement, "hide-toc");
                }
                break;
            case "hide-modnav":
                if (value === true) {
                    addClass(document.documentElement, "hide-modnav");
                } else {
                    removeClass(document.documentElement, "hide-modnav");
                }
                break;
        }
    }

    function showLightAndDark() {
        removeClass(document.getElementById("preferred-light-theme"), "hidden");
        removeClass(document.getElementById("preferred-dark-theme"), "hidden");
    }

    function hideLightAndDark() {
        addClass(document.getElementById("preferred-light-theme"), "hidden");
        addClass(document.getElementById("preferred-dark-theme"), "hidden");
    }

    function updateLightAndDark() {
        const useSystem = getSettingValue("use-system-theme");
        if (useSystem === "true" || (useSystem === null && getSettingValue("theme") === null)) {
            showLightAndDark();
        } else {
            hideLightAndDark();
        }
    }

    function setEvents(settingsElement) {
        updateLightAndDark();
        onEachLazy(settingsElement.querySelectorAll("input[type=\"checkbox\"]"), toggle => {
            const settingId = toggle.id;
            const settingValue = getSettingValue(settingId);
            if (settingValue !== null) {
                toggle.checked = settingValue === "true";
            }
            toggle.onchange = () => {
                changeSetting(toggle.id, toggle.checked);
            };
        });
        onEachLazy(settingsElement.querySelectorAll("input[type=\"radio\"]"), elem => {
            const settingId = elem.name;
            let settingValue = getSettingValue(settingId);
            if (settingId === "theme") {
                const useSystem = getSettingValue("use-system-theme");
                if (useSystem === "true" || settingValue === null) {
                    // "light" is the default theme
                    settingValue = useSystem === "false" ? "light" : "system preference";
                }
            }
            if (settingValue !== null && settingValue !== "null") {
                elem.checked = settingValue === elem.value;
            }
            elem.addEventListener("change", ev => {
                changeSetting(ev.target.name, ev.target.value);
            });
        });
    }

    /**
     * This function builds the sections inside the "settings page". It takes a `settings` list
     * as argument which describes each setting and how to render it. It returns a string
     * representing the raw HTML.
     *
     * @param {Array<Object>} settings
     *
     * @return {string}
     */
    function buildSettingsPageSections(settings) {
        let output = "";

        for (const setting of settings) {
            if (setting === "hr") {
                output += "<hr>";
                continue;
            }

            const js_data_name = setting["js_name"];
            const setting_name = setting["name"];

            if (setting["options"] !== undefined) {
                // This is a select setting.
                output += `\
<div class="setting-line" id="${js_data_name}">
    <div class="setting-radio-name">${setting_name}</div>
    <div class="setting-radio-choices">`;
                onEach(setting["options"], option => {
                    const checked = option === setting["default"] ? " checked" : "";
                    const full = `${js_data_name}-${option.replace(/ /g,"-")}`;

                    output += `\
        <label for="${full}" class="setting-radio">
            <input type="radio" name="${js_data_name}"
                id="${full}" value="${option}"${checked}>
            <span>${option}</span>
        </label>`;
                });
                output += `\
    </div>
</div>`;
            } else {
                // This is a checkbox toggle.
                const checked = setting["default"] === true ? " checked" : "";
                output += `\
<div class="setting-line">\
    <label class="setting-check">\
        <input type="checkbox" id="${js_data_name}"${checked}>\
        <span>${setting_name}</span>\
    </label>\
</div>`;
            }
        }
        return output;
    }

    /**
     * This function builds the "settings page" and returns the generated HTML element.
     *
     * @return {HTMLElement}
     */
    function buildSettingsPage() {
        const theme_names = getVar("themes").split(",").filter(t => t);
        theme_names.push("light", "dark", "ayu");

        const settings = [
            {
                "name": "Theme",
                "js_name": "theme",
                "default": "system preference",
                "options": theme_names.concat("system preference"),
            },
            {
                "name": "Preferred light theme",
                "js_name": "preferred-light-theme",
                "default": "light",
                "options": theme_names,
            },
            {
                "name": "Preferred dark theme",
                "js_name": "preferred-dark-theme",
                "default": "dark",
                "options": theme_names,
            },
            {
                "name": "Auto-hide item contents for large items",
                "js_name": "auto-hide-large-items",
                "default": true,
            },
            {
                "name": "Auto-hide item methods' documentation",
                "js_name": "auto-hide-method-docs",
                "default": false,
            },
            {
                "name": "Auto-hide trait implementation documentation",
                "js_name": "auto-hide-trait-implementations",
                "default": false,
            },
            {
                "name": "Directly go to item in search if there is only one result",
                "js_name": "go-to-only-result",
                "default": false,
            },
            {
                "name": "Show line numbers on code examples",
                "js_name": "line-numbers",
                "default": false,
            },
            {
                "name": "Hide persistent navigation bar",
                "js_name": "hide-sidebar",
                "default": false,
            },
            {
                "name": "Hide table of contents",
                "js_name": "hide-toc",
                "default": false,
            },
            {
                "name": "Hide module navigation",
                "js_name": "hide-modnav",
                "default": false,
            },
            {
                "name": "Disable keyboard shortcuts",
                "js_name": "disable-shortcuts",
                "default": false,
            },
        ];

        // Then we build the DOM.
        const elementKind = isSettingsPage ? "section" : "div";
        const innerHTML = `<div class="settings">${buildSettingsPageSections(settings)}</div>`;
        const el = document.createElement(elementKind);
        el.id = "settings";
        if (!isSettingsPage) {
            el.className = "popover";
        }
        el.innerHTML = innerHTML;

        if (isSettingsPage) {
            document.getElementById(MAIN_ID).appendChild(el);
        } else {
            el.setAttribute("tabindex", "-1");
            getSettingsButton().appendChild(el);
        }
        return el;
    }

    const settingsMenu = buildSettingsPage();

    function displaySettings() {
        settingsMenu.style.display = "";
        onEachLazy(settingsMenu.querySelectorAll("input[type='checkbox']"), el => {
            const val = getSettingValue(el.id);
            const checked = val === "true";
            if (checked !== el.checked && val !== null) {
                el.checked = checked;
            }
        });
    }

    function settingsBlurHandler(event) {
        blurHandler(event, getSettingsButton(), window.hidePopoverMenus);
    }

    if (isSettingsPage) {
        // We replace the existing "onclick" callback to do nothing if clicked.
        getSettingsButton().onclick = event => {
            event.preventDefault();
        };
    } else {
        // We replace the existing "onclick" callback.
        const settingsButton = getSettingsButton();
        const settingsMenu = document.getElementById("settings");
        settingsButton.onclick = event => {
            if (settingsMenu.contains(event.target)) {
                return;
            }
            event.preventDefault();
            const shouldDisplaySettings = settingsMenu.style.display === "none";

            window.hideAllModals();
            if (shouldDisplaySettings) {
                displaySettings();
            }
        };
        settingsButton.onblur = settingsBlurHandler;
        settingsButton.querySelector("a").onblur = settingsBlurHandler;
        onEachLazy(settingsMenu.querySelectorAll("input"), el => {
            el.onblur = settingsBlurHandler;
        });
        settingsMenu.onblur = settingsBlurHandler;
    }

    // We now wait a bit for the web browser to end re-computing the DOM...
    setTimeout(() => {
        setEvents(settingsMenu);
        // The setting menu is already displayed if we're on the settings page.
        if (!isSettingsPage) {
            displaySettings();
        }
        removeClass(getSettingsButton(), "rotate");
    }, 0);
})();
