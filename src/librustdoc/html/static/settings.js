// Local js definitions:
/* global getCurrentValue, updateLocalStorage, updateSystemTheme */

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

    function setEvents() {
        var elems = {
            toggles: document.getElementsByClassName("slider"),
            selects: document.getElementsByClassName("select-wrapper")
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

    setEvents();
})();
