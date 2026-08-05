 /* global addClass, hasClass, removeClass, onEachLazy, nonnull */

"use strict";

(function() {
    // Number of lines shown when code viewer is not expanded.
    // DEFAULT is the first example shown by default, while HIDDEN is
    // the examples hidden beneath the "More examples" toggle.
    //
    // NOTE: these values MUST be synchronized with certain rules in rustdoc.css!
    const DEFAULT_MAX_LINES = 5;
    const HIDDEN_MAX_LINES = 10;

    /**
     * Scroll code block to the given code location
     * @param {HTMLElement} elt
     * @param {[number, number]} loc
     * @param {boolean} isHidden
     */
    function scrollToLoc(elt, loc, isHidden) {
        /** @type {HTMLElement[]} */
        // blocked on https://github.com/microsoft/TypeScript/issues/29037
        // @ts-expect-error
        const lines = elt.querySelectorAll("[data-nosnippet]");
        let scrollOffset;

        // If the block is greater than the size of the viewer,
        // then scroll to the top of the block. Otherwise scroll
        // to the middle of the block.
        const maxLines = isHidden ? HIDDEN_MAX_LINES : DEFAULT_MAX_LINES;
        if (loc[1] - loc[0] > maxLines) {
            const line = Math.max(0, loc[0] - 1);
            scrollOffset = lines[line].offsetTop;
        } else {
            const halfHeight = elt.offsetHeight / 2;
            const offsetTop = lines[loc[0]].offsetTop;
            const lastLine = lines[loc[1]];
            const offsetBot = lastLine.offsetTop + lastLine.offsetHeight;
            const offsetMid = (offsetTop + offsetBot) / 2;
            scrollOffset = offsetMid - halfHeight;
        }

        nonnull(lines[0].parentElement).scrollTo(0, scrollOffset);
        nonnull(elt.querySelector(".rust")).scrollTo(0, scrollOffset);
    }

    /**
     * @param {HTMLElement} parent
     * @param {string} className
     * @param {string} content
     */
    function createScrapeButton(parent, className, content) {
        const button = document.createElement("button");
        button.className = className;
        button.title = content;
        parent.insertBefore(button, parent.firstChild);
        return button;
    }

    window.updateScrapedExample = (example, buttonHolder) => {
        let locIndex = 0;
        const highlights = Array.prototype.slice.call(example.querySelectorAll(".highlight"));

        /** @type {HTMLAnchorElement} */
        const link = nonnull(example.querySelector(".scraped-example-title a"));
        let expandButton = null;

        if (!example.classList.contains("expanded")) {
            expandButton = createScrapeButton(buttonHolder, "expand", "Show all");
        }
        const isHidden = nonnull(example.parentElement).classList.contains("more-scraped-examples");

        // @ts-expect-error
        const locs = example.locs;
        if (locs.length > 1) {
            const next = createScrapeButton(buttonHolder, "next", "Next usage");
            const prev = createScrapeButton(buttonHolder, "prev", "Previous usage");

            // Toggle through list of examples in a given file
            /** @type {function(function(): void): void} */
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

            prev.addEventListener("click", () => {
                onChangeLoc(() => {
                    locIndex = (locIndex - 1 + locs.length) % locs.length;
                });
            });

            next.addEventListener("click", () => {
                onChangeLoc(() => {
                    locIndex = (locIndex + 1) % locs.length;
                });
            });
        }

        if (expandButton) {
            expandButton.addEventListener("click", () => {
                if (hasClass(example, "expanded")) {
                    removeClass(example, "expanded");
                    removeClass(expandButton, "collapse");
                    expandButton.title = "Show all";
                    scrollToLoc(example, locs[0][0], isHidden);
                } else {
                    addClass(example, "expanded");
                    addClass(expandButton, "collapse");
                    expandButton.title = "Show single example";
                }
            });
        }
    };

    /**
     * Initialize the `locs` field
     *
     * @param {HTMLElement & {locs?: rustdoc.ScrapedLoc[]}} example
     * @param {boolean} isHidden
     */
    function setupLoc(example, isHidden) {
        const locs_str = nonnull(example.attributes.getNamedItem("data-locs")).textContent;
        const locs =
              JSON.parse(nonnull(nonnull(locs_str)));
        example.locs = locs;
        // Start with the first example in view
        scrollToLoc(example, locs[0][0], isHidden);
    }

    const firstExamples = document.querySelectorAll(".scraped-example-list > .scraped-example");
    onEachLazy(firstExamples, el => setupLoc(el, false));
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
            // visible. This is necessary since setupLoc calls scrollToLoc which
            // depends on offsetHeight, a property that requires an element to be visible to
            // compute correctly.
            setTimeout(() => {
                onEachLazy(moreExamples, el => setupLoc(el, true));
            });
        }, {once: true});
    });
})();
