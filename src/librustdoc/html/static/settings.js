(function () {
    function changeSetting(settingName, isEnabled) {
        updateLocalStorage('rustdoc-' + settingName, isEnabled);
    }

    function updateChildren(elem) {
        var state = elem.checked;

        var parentLabel = elem.parentElement;
        if (!parentLabel) {
            return;
        }
        var parentSettingLine = parentLabel.parentElement;
        if (!parentSettingLine) {
            return;
        }
        var elems = parentSettingLine.getElementsByClassName("sub-setting")[0];
        if (!elems) {
            return;
        }

        var toggles = elems.getElementsByTagName("input");
        if (!toggles || toggles.length < 1) {
            return;
        }
        var ev = new Event("change");
        for (var x = 0; x < toggles.length; ++x) {
            if (toggles[x].checked !== state) {
                toggles[x].checked = state;
                toggles[x].dispatchEvent(ev);
            }
        }
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
                updateChildren(this);
            };
        }
    }

    setEvents();
})();
