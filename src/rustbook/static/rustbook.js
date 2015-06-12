document.addEventListener("DOMContentLoaded", function(event) {

    document.getElementById("toggle-nav").onclick = toggleNav;
    function toggleNav() {
        var toc = document.getElementById("toc");
        var pagewrapper = document.getElementById("page-wrapper");
        toggleClass(toc, "mobile-hidden");
        toggleClass(pagewrapper, "mobile-hidden");
    }

    function toggleClass(el, className) {
        // from http://youmightnotneedjquery.com/
        if (el.classList) {
            el.classList.toggle(className);
        } else {
            var classes = el.className.split(' ');
            var existingIndex = classes.indexOf(className);

            if (existingIndex >= 0) {
                classes.splice(existingIndex, 1);
            } else {
                classes.push(className);
            }

            el.className = classes.join(' ');
        }
    }

    // The below code is used to add prev and next navigation links to the bottom
    // of each of the sections.
    // It works by extracting the current page based on the url and iterates over
    // the menu links until it finds the menu item for the current page. We then
    // create a copy of the preceding and following menu links and add the
    // correct css class and insert them into the bottom of the page.
    var toc = document.getElementById('toc').getElementsByTagName('a');
    var href = document.location.pathname.split('/').pop();
    if (href === 'index.html' || href === '') {
        href = 'README.html';
    }

    for (var i = 0; i < toc.length; i++) {
        if (toc[i].attributes['href'].value.split('/').pop() === href) {
            var nav = document.createElement('p');
            nav.className = 'nav-previous-next';
            if (i > 0) {
                var prevNode = toc[i-1].cloneNode(true);
                prevNode.className = 'left';
                nav.appendChild(prevNode);
            }
            if (i < toc.length - 1) {
                var nextNode = toc[i+1].cloneNode(true);
                nextNode.className = 'right';
                nav.appendChild(nextNode);
            }
            document.getElementById('page').appendChild(nav);
            break;
        }
    }

});
