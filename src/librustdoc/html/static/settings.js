(function () {
    function changeSetting(settingName, isEnabled) {
        updateLocalStorage('rustdoc-' + settingName, isEnabled);
    }

    function getSettingValue(settingName) {
        return getCurrentValue('rustdoc-' + settingName);
    }

    function setEvents() {
        var elems = document.getElementsByClassName("slider");
        if (!elems || elems.length === 0) {
            return;
        }
        for (var i = 0; i < elems.length; ++i) {
            var toggle = elems[i].previousElementSibling;
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

    setEvents();
})();
