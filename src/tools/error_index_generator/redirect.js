(function() {
    if (window.location.hash) {
        let code = window.location.hash.replace(/^#/, '');
        // We have to make sure this pattern matches to avoid inadvertently creating an
        // open redirect.
        if (!/^E[0-9]+$/.test(code)) {
            return;
        }
        if (window.location.pathname.indexOf("/error_codes/") !== -1) {
            // We're not at the top level, so we don't prepend with "./error_codes/".
            window.location = './' + code + '.html';
        } else {
            window.location = './error_codes/' + code + '.html';
        }
    }
})()
