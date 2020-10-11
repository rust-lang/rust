// Local js definitions:
/* global getCurrentValue, updateLocalStorage */

(function () {
    function changeSetting(settingName, value) {
        updateLocalStorage('rustdoc-' + settingName, value);

        switch (settingName) {
            case 'preferred-dark-theme':
            case 'preferred-light-theme':
            case 'use-system-theme':
                updateSystemTheme();
                break;
        }
    }

    function getSettingValue(settingName) {
        return getCurrentValue('rustdoc-' + settingName);
    }

    function setEvents() {
        var elems = {
            toggles: document.getElementsByClassName("slider"),
            selects: document.getElementsByClassName("select-wrapper")
        };

        if (elems.toggles && elems.toggles.length > 0) {
            for (var i = 0; i < elems.toggles.length; ++i) {
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
            for (var i = 0; i < elems.selects.length; ++i) {
                var select = elems.selects[i].getElementsByTagName('select')[0];
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
