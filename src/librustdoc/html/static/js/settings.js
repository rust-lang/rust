// Local js definitions:
/* global getSettingValue, getVirtualKey, onEachLazy, updateLocalStorage, updateSystemTheme */
/* global addClass, removeClass */

(function () {
    function changeSetting(settingName, value) {
        updateLocalStorage(settingName, value);

        switch (settingName) {
            case "theme":
            case "preferred-dark-theme":
            case "preferred-light-theme":
            case "use-system-theme":
                updateSystemTheme();
                updateLightAndDark();
                break;
        }
    }

    function handleKey(ev) {
        // Don't interfere with browser shortcuts
        if (ev.ctrlKey || ev.altKey || ev.metaKey) {
            return;
        }
        switch (getVirtualKey(ev)) {
            case "Enter":
            case "Return":
            case "Space":
                ev.target.checked = !ev.target.checked;
                ev.preventDefault();
                break;
        }
    }

    function showLightAndDark() {
        addClass(document.getElementById("theme").parentElement, "hidden");
        removeClass(document.getElementById("preferred-light-theme").parentElement, "hidden");
        removeClass(document.getElementById("preferred-dark-theme").parentElement, "hidden");
    }

    function hideLightAndDark() {
        addClass(document.getElementById("preferred-light-theme").parentElement, "hidden");
        addClass(document.getElementById("preferred-dark-theme").parentElement, "hidden");
        removeClass(document.getElementById("theme").parentElement, "hidden");
    }

    function updateLightAndDark() {
        if (getSettingValue("use-system-theme") !== "false") {
            showLightAndDark();
        } else {
            hideLightAndDark();
        }
    }

    function setEvents() {
        updateLightAndDark();
        onEachLazy(document.getElementsByClassName("slider"), function(elem) {
            var toggle = elem.previousElementSibling;
            var settingId = toggle.id;
            var settingValue = getSettingValue(settingId);
            if (settingValue !== null) {
                toggle.checked = settingValue === "true";
            }
            toggle.onchange = function() {
                changeSetting(this.id, this.checked);
            };
            toggle.onkeyup = handleKey;
            toggle.onkeyrelease = handleKey;
        });
        onEachLazy(document.getElementsByClassName("select-wrapper"), function(elem) {
            var select = elem.getElementsByTagName("select")[0];
            var settingId = select.id;
            var settingValue = getSettingValue(settingId);
            if (settingValue !== null) {
                select.value = settingValue;
            }
            select.onchange = function() {
                changeSetting(this.id, this.value);
            };
        });
        onEachLazy(document.querySelectorAll("input[type=\"radio\"]"), function(elem) {
            const settingId = elem.name;
            const settingValue = getSettingValue(settingId);
            if (settingValue !== null && settingValue !== "null") {
                elem.checked = settingValue === elem.value;
            }
            elem.addEventListener("change", function(ev) {
                changeSetting(ev.target.name, ev.target.value);
            });
        });
        document.getElementById("back").addEventListener("click", function() {
            history.back();
        });
    }

    window.addEventListener("DOMContentLoaded", setEvents);
})();
