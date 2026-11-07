for (const elem of document.querySelectorAll("pre.playground")) {
    if (elem.querySelector(".compile_fail") === null) {
        continue;
    }
    const child = document.createElement("div");
    child.className = "tooltip";
    child.textContent = "ⓘ";
    elem.appendChild(child);
}
