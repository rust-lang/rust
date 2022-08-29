(function() {{
    if (window.location.hash) {{
        let code = window.location.hash.replace(/^#/, '');
        // We have to make sure this pattern matches to avoid inadvertently creating an
        // open redirect.
        if (/^E[0-9]+$/.test(code)) {{
            window.location = './error_codes/' + code + '.html';
        }}
    }}
}})()
