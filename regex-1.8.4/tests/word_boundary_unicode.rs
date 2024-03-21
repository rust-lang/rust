// Unicode word boundaries know about Unicode characters.
// For ASCII word boundaries, the tests are precisely inverted.
matiter!(unicode1, r"\bx\b", "áxβ");
matiter!(unicode2, r"\Bx\B", "áxβ", (2, 3));

matiter!(ascii1, r"(?-u:\b)x(?-u:\b)", "áxβ", (2, 3));
