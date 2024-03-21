mat!(prefix_literal_match, r"^abc", r"abc", Some((0, 3)));
mat!(prefix_literal_nomatch, r"^abc", r"zabc", None);
mat!(one_literal_edge, r"abc", r"xxxxxab", None);
matiter!(terminates, r"a$", r"a", (0, 1));
