/* global addClass, hasClass, removeClass, onEachLazy */

"use strict";

(function() {
    // Number of lines shown when code viewer is not expanded
    const MAX_LINES = 10;

    // Scroll code block to the given code location
    function scrollToLoc(elt, loc) {
        const lines = elt.querySelector(".src-line-numbers");
        let scrollOffset;

        // If the block is greater than the size of the viewer,
        // then scroll to the top of the block. Otherwise scroll
        // to the middle of the block.
        if (loc[1] - loc[0] > MAX_LINES) {
            const line = Math.max(0, loc[0] - 1);
            scrollOffset = lines.children[line].offsetTop;
        } else {
            const wrapper = elt.querySelector(".code-wrapper");
            const halfHeight = wrapper.offsetHeight / 2;
            const offsetMid = (lines.children[loc[0]].offsetTop
                             + lines.children[loc[1]].offsetTop) / 2;
            scrollOffset = offsetMid - halfHeight;
        }

        lines.scrollTo(0, scrollOffset);
        elt.querySelector(".rust").scrollTo(0, scrollOffset);
    }

    function updateScrapedExample(example) {
        const locs = JSON.parse(example.attributes.getNamedItem("data-locs").textContent);
        let locIndex = 0;
        const highlights = Array.prototype.slice.call(example.querySelectorAll(".highlight"));
        const link = example.querySelector(".scraped-example-title a");

        if (locs.length > 1) {
            // Toggle through list of examples in a given file
            const onChangeLoc = changeIndex => {
                removeClass(highlights[locIndex], "focus");
                changeIndex();
                scrollToLoc(example, locs[locIndex][0]);
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

            example.querySelector("next")
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
                    scrollToLoc(example, locs[0][0]);
                } else {
                    addClass(example, "expanded");
                }
            });
        }

        // Start with the first example in view
        scrollToLoc(example, locs[0][0]);
    }

    const firstExamples = document.querySelectorAll(".scraped-example-list > .scraped-example");
    onEachLazy(firstExamples, updateScrapedExample);
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
                onEachLazy(moreExamples, updateScrapedExample);
            });
        }, {once: true});
    });
})();
