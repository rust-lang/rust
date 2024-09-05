/* global addClass, hasClass, removeClass, onEachLazy */

"use strict";

(function() {
    // Number of lines shown when code viewer is not expanded.
    // DEFAULT is the first example shown by default, while HIDDEN is
    // the examples hidden beneath the "More examples" toggle.
    //
    // NOTE: these values MUST be synchronized with certain rules in rustdoc.css!
    const DEFAULT_MAX_LINES = 5;
    const HIDDEN_MAX_LINES = 10;

    // Scroll code block to the given code location
    function scrollToLoc(elt, loc, isHidden) {
        const lines = elt.querySelector(".src-line-numbers > pre");
        let scrollOffset;

        // If the block is greater than the size of the viewer,
        // then scroll to the top of the block. Otherwise scroll
        // to the middle of the block.
        const maxLines = isHidden ? HIDDEN_MAX_LINES : DEFAULT_MAX_LINES;
        if (loc[1] - loc[0] > maxLines) {
            const line = Math.max(0, loc[0] - 1);
            scrollOffset = lines.children[line].offsetTop;
        } else {
            const halfHeight = elt.offsetHeight / 2;
            const offsetTop = lines.children[loc[0]].offsetTop;
            const lastLine = lines.children[loc[1]];
            const offsetBot = lastLine.offsetTop + lastLine.offsetHeight;
            const offsetMid = (offsetTop + offsetBot) / 2;
            scrollOffset = offsetMid - halfHeight;
        }

        lines.parentElement.scrollTo(0, scrollOffset);
        elt.querySelector(".rust").scrollTo(0, scrollOffset);
    }

    function updateScrapedExample(example, isHidden) {
        const locs = JSON.parse(example.attributes.getNamedItem("data-locs").textContent);
        let locIndex = 0;
        const highlights = Array.prototype.slice.call(example.querySelectorAll(".highlight"));
        const link = example.querySelector(".scraped-example-title a");

        if (locs.length > 1) {
            // Toggle through list of examples in a given file
            const onChangeLoc = changeIndex => {
                removeClass(highlights[locIndex], "focus");
                changeIndex();
                scrollToLoc(example, locs[locIndex][0], isHidden);
                addClass(highlights[locIndex], "focus");

                const url = locs[locIndex][1];
                const title = locs[locIndex][2];

                link.href = url;
                link.innerHTML = title;
            };

            example.querySelector(".prev")
                .addEventListener("click", () => {
                    onChangeLoc(() => {
                        locIndex = (locIndex - 1 + locs.length) % locs.length;
                    });
                });

            example.querySelector(".next")
                .addEventListener("click", () => {
                    onChangeLoc(() => {
                        locIndex = (locIndex + 1) % locs.length;
                    });
                });
        }

        const expandButton = example.querySelector(".expand");
        if (expandButton) {
            expandButton.addEventListener("click", () => {
                if (hasClass(example, "expanded")) {
                    removeClass(example, "expanded");
                    scrollToLoc(example, locs[0][0], isHidden);
                } else {
                    addClass(example, "expanded");
                }
            });
        }

        // Start with the first example in view
        scrollToLoc(example, locs[0][0], isHidden);
    }

    const firstExamples = document.querySelectorAll(".scraped-example-list > .scraped-example");
    onEachLazy(firstExamples, el => updateScrapedExample(el, false));
    onEachLazy(document.querySelectorAll(".more-examples-toggle"), toggle => {
        // Allow users to click the left border of the <details> section to close it,
        // since the section can be large and finding the [+] button is annoying.
        onEachLazy(toggle.querySelectorAll(".toggle-line, .hide-more"), button => {
            button.addEventListener("click", () => {
                toggle.open = false;
            });
        });

        const moreExamples = toggle.querySelectorAll(".scraped-example");
        toggle.querySelector("summary").addEventListener("click", () => {
            // Wrapping in setTimeout ensures the update happens after the elements are actually
            // visible. This is necessary since updateScrapedExample calls scrollToLoc which
            // depends on offsetHeight, a property that requires an element to be visible to
            // compute correctly.
            setTimeout(() => {
                onEachLazy(moreExamples, el => updateScrapedExample(el, true));
            });
        }, {once: true});
    });
})();
