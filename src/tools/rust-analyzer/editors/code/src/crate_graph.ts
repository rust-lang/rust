/// <reference lib="dom" />
// @ts-expect-error d3 is untyped
import { select, transition } from "d3";
import "d3-graphviz";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const d3Select: any = select;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const d3Transition: any = transition;

const dotElement = document.getElementById("graph-data");
if (dotElement?.textContent) {
    const dot = JSON.parse(dotElement.textContent) as string;
    const graph = d3Select("#graph")
        .graphviz({ useWorker: false, useSharedWorker: false })
        .fit(true)
        .zoomScaleExtent([0.1, Infinity])
        .renderDot(dot);

    d3Select(window).on("click", (event: MouseEvent) => {
        if (event.ctrlKey) {
            graph.resetZoom(d3Transition().duration(100));
        }
    });
    d3Select(window).on("copy", (event: ClipboardEvent) => {
        event.clipboardData?.setData("text/plain", dot);
        event.preventDefault();
    });
}
