// Local js definitions:
/* global getCurrentValue, getVirtualKey, updateLocalStorage, updateSystemTheme */

(function () {
    function changeSetting(settingName, value) {
        updateLocalStorage("rustdoc-" + settingName, value);

        switch (settingName) {
            case "preferred-dark-theme":
            case "preferred-light-theme":
            case "use-system-theme":
                updateSystemTheme();
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

    function setEvents() {
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
    }

    window.addEventListener("DOMContentLoaded", setEvents);
})();
