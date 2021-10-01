/* global addClass, hasClass, removeClass, onEach */

(function () {
    // Scroll code block to put the given code location in the middle of the viewer
    function scrollToLoc(elt, loc) {
        var wrapper = elt.querySelector(".code-wrapper");
        var halfHeight = wrapper.offsetHeight / 2;
        var lines = elt.querySelector('.line-numbers');
        var offsetMid = (lines.children[loc[0]].offsetTop
                         + lines.children[loc[1]].offsetTop) / 2;
        var scrollOffset = offsetMid - halfHeight;
        lines.scrollTo(0, scrollOffset);
        elt.querySelector(".rust").scrollTo(0, scrollOffset);
    }

    function updateScrapedExample(example) {
        var locs = JSON.parse(example.attributes.getNamedItem("data-locs").textContent);
        var offset = parseInt(example.attributes.getNamedItem("data-offset").textContent);

        var locIndex = 0;
        var highlights = example.querySelectorAll('.highlight');
        var link = example.querySelector('.scraped-example-title a');
        addClass(highlights[0], 'focus');
        if (locs.length > 1) {
            // Toggle through list of examples in a given file
            var onChangeLoc = function(f) {
                removeClass(highlights[locIndex], 'focus');
                f();
                scrollToLoc(example, locs[locIndex]);
                addClass(highlights[locIndex], 'focus');

                var curLoc = locs[locIndex];
                var minLine = curLoc[0] + offset + 1;
                var maxLine = curLoc[1] + offset + 1;

                var text;
                var anchor;
                if (minLine == maxLine) {
                    text = 'line ' + minLine.toString();
                    anchor = minLine.toString();
                } else {
                    var range = minLine.toString() + '-' + maxLine.toString();
                    text = 'lines ' + range;
                    anchor = range;
                }

                var url = new URL(link.href);
                url.hash = anchor;

                link.href = url.toString();
                link.innerHTML = text;
            };

            example.querySelector('.prev')
                .addEventListener('click', function() {
                    onChangeLoc(function() {
                        locIndex = (locIndex - 1 + locs.length) % locs.length;
                    });
                });

            example.querySelector('.next')
                .addEventListener('click', function() {
                    onChangeLoc(function() { locIndex = (locIndex + 1) % locs.length; });
                });
        } else {
            // Remove buttons if there's only one example in the file
            example.querySelector('.prev').remove();
            example.querySelector('.next').remove();
        }

        var codeEl = example.querySelector('.rust');
        var codeOverflows = codeEl.scrollHeight > codeEl.clientHeight;
        var expandButton = example.querySelector('.expand');
        if (codeOverflows) {
            // If file is larger than default height, give option to expand the viewer
            expandButton.addEventListener('click', function () {
                if (hasClass(example, "expanded")) {
                    removeClass(example, "expanded");
                    scrollToLoc(example, locs[0]);
                } else {
                    addClass(example, "expanded");
                }
            });
        } else {
            // Otherwise remove expansion buttons
            addClass(example, 'expanded');
            expandButton.remove();
        }

        // Start with the first example in view
        scrollToLoc(example, locs[0]);
    }

    var firstExamples = document.querySelectorAll('.scraped-example-list > .scraped-example');
    onEach(firstExamples, updateScrapedExample);
    onEach(document.querySelectorAll('.more-examples-toggle'), function(toggle) {
        // Allow users to click the left border of the <details> section to close it,
        // since the section can be large and finding the [+] button is annoying.
        toggle.querySelector('.toggle-line').addEventListener('click', function() {
            toggle.open = false;
        });

        var moreExamples = toggle.querySelectorAll('.scraped-example');
        toggle.querySelector('summary').addEventListener('click', function() {
            // Wrapping in setTimeout ensures the update happens after the elements are actually
            // visible. This is necessary since updateScrapedExample calls scrollToLoc which
            // depends on offsetHeight, a property that requires an element to be visible to
            // compute correctly.
            setTimeout(function() { onEach(moreExamples, updateScrapedExample); });
        }, {once: true});
    });
})();
