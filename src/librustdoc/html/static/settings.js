/*!
 * Copyright 2018 The Rust Project Developers. See the COPYRIGHT
 * file at the top-level directory of this distribution and at
 * http://rust-lang.org/COPYRIGHT.
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
 * http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.
 */

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
