// Inspired by https://github.com/JorelAli/mdBook-pagetoc/tree/98ee241 (under WTFPL)

let activeHref = location.href;
function updatePageToc(elem = undefined) {
    let selectedPageTocElem = elem;
    const pagetoc = document.getElementById("pagetoc");

    function getRect(element) {
        return element.getBoundingClientRect();
    }

    function overflowTop(container, element) {
        return getRect(container).top - getRect(element).top;
    }

    function overflowBottom(container, element) {
        return getRect(container).bottom - getRect(element).bottom;
    }

    // We've not selected a heading to highlight, and the URL needs updating
    // so we need to find a heading based on the URL
    if (selectedPageTocElem === undefined && location.href !== activeHref) {
        activeHref = location.href;
        for (const pageTocElement of pagetoc.children) {
            if (pageTocElement.href === activeHref) {
                selectedPageTocElem = pageTocElement;
            }
        }
    }

    // We still don't have a selected heading, let's try and find the most
    // suitable heading based on the scroll position
    if (selectedPageTocElem === undefined) {
        const margin = window.innerHeight / 3;

        const headers = document.getElementsByClassName("header");
        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            if (selectedPageTocElem === undefined && getRect(header).top >= 0) {
                if (getRect(header).top < margin) {
                    selectedPageTocElem = header;
                } else {
                    selectedPageTocElem = headers[Math.max(0, i - 1)];
                }
            }
            // a very long last section's heading is over the screen
            if (selectedPageTocElem === undefined && i === headers.length - 1) {
                selectedPageTocElem = header;
            }
        }
    }

    // Remove the active flag from all pagetoc elements
    for (const pageTocElement of pagetoc.children) {
        pageTocElement.classList.remove("active");
    }

    // If we have a selected heading, set it to active and scroll to it
    if (selectedPageTocElem !== undefined) {
        for (const pageTocElement of pagetoc.children) {
            if (selectedPageTocElem.href.localeCompare(pageTocElement.href) === 0) {
                pageTocElement.classList.add("active");
                if (overflowTop(pagetoc, pageTocElement) > 0) {
                    pagetoc.scrollTop = pageTocElement.offsetTop;
                }
                if (overflowBottom(pagetoc, pageTocElement) < 0) {
                    pagetoc.scrollTop -= overflowBottom(pagetoc, pageTocElement);
                }
            }
        }
    }
}

if (document.getElementById("sidetoc") === null &&
    document.getElementsByClassName("header").length > 0) {
    // The sidetoc element doesn't exist yet, let's create it

    // Create the empty sidetoc and pagetoc elements
    const sidetoc = document.createElement("div");
    const pagetoc = document.createElement("div");
    sidetoc.id = "sidetoc";
    pagetoc.id = "pagetoc";
    sidetoc.appendChild(pagetoc);

    // And append them to the current DOM
    const main = document.querySelector('main');
    main.insertBefore(sidetoc, main.firstChild);

    // Populate sidebar on load
    window.addEventListener("load", () => {
        for (const header of document.getElementsByClassName("header")) {
            const link = document.createElement("a");
            link.innerHTML = header.innerHTML;
            link.href = header.hash;
            link.classList.add("pagetoc-" + header.parentElement.tagName);
            document.getElementById("pagetoc").appendChild(link);
            link.onclick = () => updatePageToc(link);
        }
        updatePageToc();
    });

    // Update page table of contents selected heading on scroll
    window.addEventListener("scroll", () => updatePageToc());
}
