/* global addClass, hasClass, removeClass, onEach */

(function () {
    // Number of lines shown when code viewer is not expanded
    const MAX_LINES = 10;

    // Scroll code block to the given code location
    function scrollToLoc(elt, loc) {
        var lines = elt.querySelector('.line-numbers');
        var scrollOffset;

        // If the block is greater than the size of the viewer,
        // then scroll to the top of the block. Otherwise scroll
        // to the middle of the block.
        if (loc[1] - loc[0] > MAX_LINES) {
            var line = Math.max(0, loc[0] - 1);
            scrollOffset = lines.children[line].offsetTop;
        } else {
            var wrapper = elt.querySelector(".code-wrapper");
            var halfHeight = wrapper.offsetHeight / 2;
            var offsetMid = (lines.children[loc[0]].offsetTop
                             + lines.children[loc[1]].offsetTop) / 2;
            scrollOffset = offsetMid - halfHeight;
        }

        lines.scrollTo(0, scrollOffset);
        elt.querySelector(".rust").scrollTo(0, scrollOffset);
    }

    function updateScrapedExample(example) {
        var locs = JSON.parse(example.attributes.getNamedItem("data-locs").textContent);
        var locIndex = 0;
        var highlights = example.querySelectorAll('.highlight');
        var link = example.querySelector('.scraped-example-title a');

        if (locs.length > 1) {
            // Toggle through list of examples in a given file
            var onChangeLoc = function(changeIndex) {
                removeClass(highlights[locIndex], 'focus');
                changeIndex();
                scrollToLoc(example, locs[locIndex][0]);
                addClass(highlights[locIndex], 'focus');

                var url = locs[locIndex][1];
                var title = locs[locIndex][2];

                link.href = url;
                link.innerHTML = title;
            };

            example.querySelector('.prev')
                .addEventListener('click', function() {
                    onChangeLoc(function() {
                        locIndex = (locIndex - 1 + locs.length) % locs.length;
                    });
                });

            example.querySelector('.next')
                .addEventListener('click', function() {
                    onChangeLoc(function() {
                        locIndex = (locIndex + 1) % locs.length;
                    });
                });
        }

        var expandButton = example.querySelector('.expand');
        if (expandButton) {
            expandButton.addEventListener('click', function () {
                if (hasClass(example, "expanded")) {
                    removeClass(example, "expanded");
                    scrollToLoc(example, locs[0][0]);
                } else {
                    addClass(example, "expanded");
                }
            });
        }

        // Start with the first example in view
        scrollToLoc(example, locs[0][0]);
    }

    var firstExamples = document.querySelectorAll('.scraped-example-list > .scraped-example');
    onEach(firstExamples, updateScrapedExample);
    onEach(document.querySelectorAll('.more-examples-toggle'), function(toggle) {
        // Allow users to click the left border of the <details> section to close it,
        // since the section can be large and finding the [+] button is annoying.
        toggle.querySelectorAll('.toggle-line, .hide-more').forEach(button => {
            button.addEventListener('click', function() {
                toggle.open = false;
            });
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
