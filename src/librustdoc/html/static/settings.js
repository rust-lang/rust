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
        var elems = {
            toggles: Array.prototype.slice.call(document.getElementsByClassName("slider")),
            selects: Array.prototype.slice.call(document.getElementsByClassName("select-wrapper")),
        };
        var i;

        if (elems.toggles && elems.toggles.length > 0) {
            for (i = 0; i < elems.toggles.length; ++i) {
                var toggle = elems.toggles[i].previousElementSibling;
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
            }
        }

        if (elems.selects && elems.selects.length > 0) {
            for (i = 0; i < elems.selects.length; ++i) {
                var select = elems.selects[i].getElementsByTagName("select")[0];
                var settingId = select.id;
                var settingValue = getSettingValue(settingId);
                if (settingValue !== null) {
                    select.value = settingValue;
                }
                select.onchange = function() {
                    changeSetting(this.id, this.value);
                };
            }
        }
    }

    window.addEventListener("DOMContentLoaded", setEvents);
})();
