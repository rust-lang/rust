pub(crate) fn urlencode(input: &str) -> String {
    input
        .replace(" ", "+")
        .replace("&", "%26")
        .replace("<", "%3C")
        .replace(">", "%3E")
        .replace("[", "%5B")
        .replace("]", "%5D")
}
