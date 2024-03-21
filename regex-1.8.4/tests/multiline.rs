matiter!(
    match_multi_1,
    r"(?m)^[a-z]+$",
    "abc\ndef\nxyz",
    (0, 3),
    (4, 7),
    (8, 11)
);
matiter!(match_multi_2, r"(?m)^$", "abc\ndef\nxyz");
matiter!(match_multi_3, r"(?m)^", "abc\ndef\nxyz", (0, 0), (4, 4), (8, 8));
matiter!(match_multi_4, r"(?m)$", "abc\ndef\nxyz", (3, 3), (7, 7), (11, 11));
matiter!(
    match_multi_5,
    r"(?m)^[a-z]",
    "abc\ndef\nxyz",
    (0, 1),
    (4, 5),
    (8, 9)
);
matiter!(match_multi_6, r"(?m)[a-z]^", "abc\ndef\nxyz");
matiter!(
    match_multi_7,
    r"(?m)[a-z]$",
    "abc\ndef\nxyz",
    (2, 3),
    (6, 7),
    (10, 11)
);
matiter!(match_multi_8, r"(?m)$[a-z]", "abc\ndef\nxyz");
matiter!(match_multi_9, r"(?m)^$", "", (0, 0));

matiter!(
    match_multi_rep_1,
    r"(?m)(?:^$)*",
    "a\nb\nc",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
);
matiter!(
    match_multi_rep_2,
    r"(?m)(?:^|a)+",
    "a\naaa\n",
    (0, 0),
    (2, 2),
    (3, 5),
    (6, 6)
);
matiter!(
    match_multi_rep_3,
    r"(?m)(?:^|a)*",
    "a\naaa\n",
    (0, 1),
    (2, 5),
    (6, 6)
);
matiter!(
    match_multi_rep_4,
    r"(?m)(?:^[a-z])+",
    "abc\ndef\nxyz",
    (0, 1),
    (4, 5),
    (8, 9)
);
matiter!(
    match_multi_rep_5,
    r"(?m)(?:^[a-z]{3}\n?)+",
    "abc\ndef\nxyz",
    (0, 11)
);
matiter!(
    match_multi_rep_6,
    r"(?m)(?:^[a-z]{3}\n?)*",
    "abc\ndef\nxyz",
    (0, 11)
);
matiter!(
    match_multi_rep_7,
    r"(?m)(?:\n?[a-z]{3}$)+",
    "abc\ndef\nxyz",
    (0, 11)
);
matiter!(
    match_multi_rep_8,
    r"(?m)(?:\n?[a-z]{3}$)*",
    "abc\ndef\nxyz",
    (0, 11)
);
matiter!(
    match_multi_rep_9,
    r"(?m)^*",
    "\naa\n",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4)
);
matiter!(match_multi_rep_10, r"(?m)^+", "\naa\n", (0, 0), (1, 1), (4, 4));
matiter!(
    match_multi_rep_11,
    r"(?m)$*",
    "\naa\n",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4)
);
matiter!(match_multi_rep_12, r"(?m)$+", "\naa\n", (0, 0), (3, 3), (4, 4));
matiter!(match_multi_rep_13, r"(?m)(?:$\n)+", "\n\naaa\n\n", (0, 2), (5, 7));
matiter!(
    match_multi_rep_14,
    r"(?m)(?:$\n)*",
    "\n\naaa\n\n",
    (0, 2),
    (3, 3),
    (4, 4),
    (5, 7)
);
matiter!(match_multi_rep_15, r"(?m)(?:$\n^)+", "\n\naaa\n\n", (0, 2), (5, 7));
matiter!(
    match_multi_rep_16,
    r"(?m)(?:^|$)+",
    "\n\naaa\n\n",
    (0, 0),
    (1, 1),
    (2, 2),
    (5, 5),
    (6, 6),
    (7, 7)
);
matiter!(
    match_multi_rep_17,
    r"(?m)(?:$\n)*",
    "\n\naaa\n\n",
    (0, 2),
    (3, 3),
    (4, 4),
    (5, 7)
);
